"""Document AI client + processor lookup.

The processor type id is discovered with fetch_processor_types() rather than hardcoded,
so a rename upstream surfaces as "no layout processor available here" instead of a 400.
"""

from google.api_core.client_options import ClientOptions

# Layout Parser's `document_layout` field exists only on the v1beta3 Document proto --
# v1 has `pages` but no DocumentLayout, so a v1 client silently returns nothing usable.
from google.cloud import documentai_v1beta3 as documentai

from .config import PROCESSOR_TYPE_MATCH


def enable_api(project, service="documentai.googleapis.com"):
    """Turn the API on with the ADC credentials, since the gcloud CLI has no logged-in
    account here. Enabling costs nothing by itself -- billing starts on use -- and is
    reversible with `gcloud services disable`. Returns the API's JSON response.
    """
    import json
    import urllib.error
    import urllib.request

    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    url = f"https://serviceusage.googleapis.com/v1/projects/{project}" f"/services/{service}:enable"
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"enable {service} failed ({error.code}): {error.read().decode()[:500]}"
        ) from error


def make_client(location):
    return documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )


def processor_type(client, project, location, match=PROCESSOR_TYPE_MATCH):
    parent = client.common_location_path(project, location)
    types = client.fetch_processor_types(parent=parent).processor_types
    hits = [t for t in types if match in t.type_.lower()]
    if not hits:
        raise SystemExit(
            f"no {match!r} processor type in {location}; available: "
            + ", ".join(sorted(t.type_ for t in types))[:400]
        )
    for t in hits:  # prefer the plain one over variants
        if t.type_.upper().startswith(match.upper()):
            return t.type_
    return hits[0].type_


def get_or_create(
    client,
    project,
    location,
    display_name="figure-gt-layout-parser",
    processor_id=None,
    create=True,
    type_match=PROCESSOR_TYPE_MATCH,
):
    """-> processor resource name. Reuses an existing processor before making one.

    Creating a processor is a change to someone's cloud project, so `create=False` turns
    this into a lookup that fails loudly rather than provisioning behind their back.
    """
    parent = client.common_location_path(project, location)
    if processor_id:
        return client.processor_path(project, location, processor_id)

    want = processor_type(client, project, location, type_match)
    for p in client.list_processors(parent=parent).processors:
        if p.type_ == want:
            return p.name
    if not create:
        raise SystemExit(f"no {want} processor in {project}/{location} and create=False")
    proc = client.create_processor(
        parent=parent, processor=documentai.Processor(type_=want, display_name=display_name)
    )
    return proc.name


def process_pdf(client, name, pdf_bytes):
    """One synchronous call. Caller must keep the doc within the 15-page / 20 MB limit."""
    req = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf"),
    )
    return client.process_document(request=req).document
