# Retrieval Benchmark Query Audit Sheet

This audit sheet contains all 110 grounded retrieval tasks for human quality verification.

- **Development Queries**: 60 single-page + 20 multi-page across 10 regions (80 grounded) + 5 out-of-corpus negatives.
- **Final Test Queries**: 15 single-page + 5 multi-page (20 grounded) + 5 out-of-corpus negatives.

| ID | Split | Type | Region | Gold Pages | Question | Verification Status |
|---|---|---|---|---|---|:---:|
| `q_dev_01` | `dev` | `single_page` | `p0001-p0100` | `p0035` | What is the longest bone in the human body, and how is its articulation described? | [x] Verified |
| `q_dev_02` | `dev` | `single_page` | `p0001-p0100` | `p0051` | Why does the liver receive two distinct kinds of blood, and what is each used for? | [x] Verified |
| `q_dev_03` | `dev` | `single_page` | `p0001-p0100` | `p0027` | What three elementary microscopic structures comprise an animal cell according to physiological anatomy? | [x] Verified |
| `q_dev_04` | `dev` | `single_page` | `p0001-p0100` | `p0056` | What is the vital physiological office of absorption in sustaining the human body? | [x] Verified |
| `q_dev_05` | `dev` | `single_page` | `p0001-p0100` | `p0071` | Which organs constitute the human respiratory system, and what is the function of the larynx? | [x] Verified |
| `q_dev_06` | `dev` | `single_page` | `p0001-p0100` | `p0092` | Where are the kidneys located in the human body, and how is their anatomical shape configured? | [x] Verified |
| `q_dev_07` | `dev` | `single_page` | `p0101-p0200` | `p0101` | What is the nature and role of the spinal cord in relation to the brain and involuntary impulses? | [x] Verified |
| `q_dev_08` | `dev` | `single_page` | `p0101-p0200` | `p0109` | To which internal organs is the tenth pair of cranial nerves (the pneumogastric or par vagum) distributed? | [x] Verified |
| `q_dev_09` | `dev` | `single_page` | `p0101-p0200` | `p0114` | How is the outer protective tunic of the human eyeball anatomically constructed? | [x] Verified |
| `q_dev_10` | `dev` | `single_page` | `p0101-p0200` | `p0117` | What physical atmospheric mechanism enables the perception of hearing, and into what divisions is the ear separated? | [x] Verified |
| `q_dev_11` | `dev` | `single_page` | `p0101-p0200` | `p0120` | What physiological conditions are required for the tongue to perceive the taste of a substance? | [x] Verified |
| `q_dev_12` | `dev` | `single_page` | `p0101-p0200` | `p0165` | What cranial development and physiological qualities characterize the lymphatic temperament? | [x] Verified |
| `q_dev_13` | `dev` | `single_page` | `p0201-p0300` | `p0218` | What physiological phenomena mark female puberty, and what is the biological origin and duration of the menses? | [x] Verified |
| `q_dev_14` | `dev` | `single_page` | `p0201-p0300` | `p0224` | What vascular structure develops between the uterus and embryo after conception, and what respiratory function does it perform? | [x] Verified |
| `q_dev_15` | `dev` | `single_page` | `p0201-p0300` | `p0227` | What maternal bodily symptoms indicate pregnancy, and how is the term of gestation computed? | [x] Verified |
| `q_dev_16` | `dev` | `single_page` | `p0201-p0300` | `p0233` | What constitutes the proper ventilation of a school room according to hygiene authorities? | [x] Verified |
| `q_dev_17` | `dev` | `single_page` | `p0201-p0300` | `p0238` | How does intense illumination reveal the presence of suspended organic particles in atmospheric air? | [x] Verified |
| `q_dev_18` | `dev` | `single_page` | `p0201-p0300` | `p0286` | Why is sleep an indispensable physiological necessity for the human organism, and what restorative effects does it produce? | [x] Verified |
| `q_dev_19` | `dev` | `single_page` | `p0301-p0400` | `p0308` | What general rules govern medicinal dosage according to the age, sex, and constitutional susceptibility of the patient? | [x] Verified |
| `q_dev_20` | `dev` | `single_page` | `p0301-p0400` | `p0312` | What botanical plant is known as May-apple, and what is the proper dosage for its active principle, Podophyllin? | [x] Verified |
| `q_dev_21` | `dev` | `single_page` | `p0301-p0400` | `p0316` | What are the therapeutic properties and physiological action of Dr. Pierce's Golden Medical Discovery? | [x] Verified |
| `q_dev_22` | `dev` | `single_page` | `p0301-p0400` | `p0328` | How are astringent medicines defined in materia medica, and what physiological sensation and effect do they produce on tissues? | [x] Verified |
| `q_dev_23` | `dev` | `single_page` | `p0301-p0400` | `p0347` | How do emetics function to evacuate the stomach, and what distinct roles are played by irritant vs systemic agents? | [x] Verified |
| `q_dev_24` | `dev` | `single_page` | `p0301-p0400` | `p0360` | What are the medicinal properties of Golden Seal (Hydrastis Canadensis) in 19th-century therapeutics? | [x] Verified |
| `q_dev_25` | `dev` | `single_page` | `p0401-p0500` | `p0403` | What is the normal ratio of pulse beats to respirations in health, and at what frequency does the heart rate indicate danger in adults? | [x] Verified |
| `q_dev_26` | `dev` | `single_page` | `p0401-p0500` | `p0406` | What pathological definition and four cardinal signs characterize the state of inflammation? | [x] Verified |
| `q_dev_27` | `dev` | `single_page` | `p0401-p0500` | `p0413` | What gaseous cause produces malarial fevers, and in what terrain is it predominantly generated? | [x] Verified |
| `q_dev_28` | `dev` | `single_page` | `p0401-p0500` | `p0422` | What constitutional features and local throat manifestations characterize diphtheria? | [x] Verified |
| `q_dev_29` | `dev` | `single_page` | `p0401-p0500` | `p0440` | What is the chemical composition of tubercular matter, and how does it form in scrofulous diathesis? | [x] Verified |
| `q_dev_30` | `dev` | `single_page` | `p0401-p0500` | `p0486` | What pathological changes occur in the mucous membrane of the nasal passages during chronic nasal catarrh (ozæna)? | [x] Verified |
| `q_dev_31` | `dev` | `single_page` | `p0501-p0600` | `p0503` | How does spasmodic croup clinically differ from true membranous croup in its pathology and febrile symptoms? | [x] Verified |
| `q_dev_32` | `dev` | `single_page` | `p0501-p0600` | `p0515` | What dietary principles and food selections are recommended to maintain nutrition and arrest tissue waste in consumptive invalids? | [x] Verified |
| `q_dev_33` | `dev` | `single_page` | `p0501-p0600` | `p0530` | What physical signs and pathological stages characterize acute pleurisy from initial chill to pleural effusion? | [x] Verified |
| `q_dev_34` | `dev` | `single_page` | `p0501-p0600` | `p0541` | What clinical features and organic cardiac lesions are associated with angina pectoris (neuralgia of the heart)? | [x] Verified |
| `q_dev_35` | `dev` | `single_page` | `p0501-p0600` | `p0550` | How can gastralgia (neuralgia of the stomach) be clinically differentiated from inflammatory gastritis by administering a diagnostic agent? | [x] Verified |
| `q_dev_36` | `dev` | `single_page` | `p0501-p0600` | `p0572` | What specific saline cathartic compound and systemic eliminative treatment are prescribed for painters suffering from lead colic? | [x] Verified |
| `q_dev_37` | `dev` | `single_page` | `p0601-p0700` | `p0603` | Why are excision with scissors and application of red-hot irons or caustics considered hazardous methods for treating large hemorrhoids? | [x] Verified |
| `q_dev_38` | `dev` | `single_page` | `p0601-p0700` | `p0610` | What three distinct structural forms of anal fistula (fistula in ano) occur in medical practice? | [x] Verified |
| `q_dev_39` | `dev` | `single_page` | `p0601-p0700` | `p0625` | What physical changes occur in the paralyzed limbs and skeletal structures of children suffering from infantile paralysis? | [x] Verified |
| `q_dev_40` | `dev` | `single_page` | `p0601-p0700` | `p0635` | What premonitory sensory warning (aura) heralds an epileptic seizure, and how can inhalation of nitrite of amyl abort it? | [x] Verified |
| `q_dev_41` | `dev` | `single_page` | `p0601-p0700` | `p0639` | How do the uncontrollable muscular contractions in chorea (St. Vitus' dance) behave during wakefulness versus during sleep? | [x] Verified |
| `q_dev_42` | `dev` | `single_page` | `p0601-p0700` | `p0676` | What clinical technique is used to extract foreign particles lodged underneath the upper eyelid? | [x] Verified |
| `q_dev_43` | `dev` | `single_page` | `p0701-p0800` | `p0703` | What topical applications and alterative regimens are recommended for relieving and curing eczematous eruptions? | [x] Verified |
| `q_dev_44` | `dev` | `single_page` | `p0701-p0800` | `p0718` | Where do felons (whitlows) anatomically originate, and why does their deep seated nature threaten tendons and bone? | [x] Verified |
| `q_dev_45` | `dev` | `single_page` | `p0701-p0800` | `p0723` | What congenital anomalies and mechanical uterine obstructions can prevent the natural exit of the menstrual flow in young women? | [x] Verified |
| `q_dev_46` | `dev` | `single_page` | `p0701-p0800` | `p0737` | What is the etymology and clinical definition of menorrhagia, and how does it affect systemic vitality? | [x] Verified |
| `q_dev_47` | `dev` | `single_page` | `p0701-p0800` | `p0745` | How do vaginal and uterine leucorrheal discharges chemically and microscopically differ from each other? | [x] Verified |
| `q_dev_48` | `dev` | `single_page` | `p0701-p0800` | `p0799` | What strict nutritional diet and bathing regimen are prescribed for managing patients suffering from diabetes? | [x] Verified |
| `q_dev_49` | `dev` | `single_page` | `p0801-p0900` | `p0801` | What mechanical obstructions give rise to retention of urine, and how does this condition differ from renal suppression? | [x] Verified |
| `q_dev_50` | `dev` | `single_page` | `p0801-p0900` | `p0805` | What is chronic cystitis (vesical catarrh), and what soothing mucilaginous diuretics are administered following acute symptoms? | [x] Verified |
| `q_dev_51` | `dev` | `single_page` | `p0801-p0900` | `p0813` | What are the relative indications and historical mortality rates for median versus lateral section lithotomy in stone removal? | [x] Verified |
| `q_dev_52` | `dev` | `single_page` | `p0801-p0900` | `p0817` | What conservative instrumentation and internal surgical operations (urethrotomy) are employed for relieving organic urethral stricture? | [x] Verified |
| `q_dev_53` | `dev` | `single_page` | `p0801-p0900` | `p0877` | What degenerative changes in the testicle frequently accompany advanced varicocele and complicate seminal debility? | [x] Verified |
| `q_dev_54` | `dev` | `single_page` | `p0801-p0900` | `p0894` | Under what clinical circumstances does acute orchitis (inflammation of the testicles) develop as a secondary affection? | [x] Verified |
| `q_dev_55` | `dev` | `single_page` | `p0901-p1034` | `p0902` | Why must primary reliance be placed on constitutional alteratives and potassium iodide in combating syphilitic infection? | [x] Verified |
| `q_dev_56` | `dev` | `single_page` | `p0901-p1034` | `p0907` | What anatomical characteristics make inguinal hernia the most common variety of rupture in adult males? | [x] Verified |
| `q_dev_57` | `dev` | `single_page` | `p0901-p1034` | `p0913` | How does rapid evaporation from an ether or rhigolene spray apparatus produce momentary local anæsthesia for minor surgical operations? | [x] Verified |
| `q_dev_58` | `dev` | `single_page` | `p0901-p1034` | `p0915` | How can emergency arterial hemorrhage be arrested using a field tourniquet improvised from a handkerchief or by acute joint flexion? | [x] Verified |
| `q_dev_59` | `dev` | `single_page` | `p0901-p1034` | `p0919` | What postural rotation movements and back pressure techniques are performed to restore breathing in apparent drowning? | [x] Verified |
| `q_dev_60` | `dev` | `single_page` | `p0901-p1034` | `p0922` | What specific chemical antidotes and emergency emetics neutralize acute poisoning from copper vitriol, lead compounds, and corrosive sublimate? | [x] Verified |
| `q_multi_dev_01` | `dev` | `multi_page` | `p0001-p0100` | `p0045, p0046` | Which primary anatomical organs constitute the human digestive system, and how are the permanent teeth classified? | [x] Verified |
| `q_multi_dev_02` | `dev` | `multi_page` | `p0001-p0100` | `p0064, p0065, p0066` | How is the human heart structured into cavities and partitions, and how does blood circulate through the auricles and ventricles? | [x] Verified |
| `q_multi_dev_03` | `dev` | `multi_page` | `p0101-p0200` | `p0104, p0103, p0105` | What are the anatomical divisions of the human brain, and where are the corpora pyramidalia located in the medulla oblongata? | [x] Verified |
| `q_multi_dev_04` | `dev` | `multi_page` | `p0101-p0200` | `p0185, p0186` | What physical characteristics and mental tendencies distinguish individuals possessing the encephalic temperament? | [x] Verified |
| `q_multi_dev_05` | `dev` | `multi_page` | `p0201-p0300` | `p0237, p0238` | How does solar illumination reveal airborne dust, and what prophylactic method was suggested to intercept noxious particulate matter? | [x] Verified |
| `q_multi_dev_06` | `dev` | `multi_page` | `p0201-p0300` | `p0286, p0287` | What physiological restorative benefits occur during sleep, and what hygienic recommendations are given regarding bedding materials? | [x] Verified |
| `q_multi_dev_07` | `dev` | `multi_page` | `p0301-p0400` | `p0308, p0309` | How are medical remedies classified according to their physiological properties, and for what age group are general handbook doses specified? | [x] Verified |
| `q_multi_dev_08` | `dev` | `multi_page` | `p0301-p0400` | `p0389, p0390` | What nutritional principles govern diet in inflammatory illnesses, and how is pure beef tea prepared by closed-vessel water-bath extraction? | [x] Verified |
| `q_multi_dev_09` | `dev` | `multi_page` | `p0401-p0500` | `p0413, p0414` | What therapeutic measures are recommended during the successive cold, hot, and sweating stages of an intermittent fever paroxysm to prevent recurrence? | [x] Verified |
| `q_multi_dev_10` | `dev` | `multi_page` | `p0401-p0500` | `p0447, p0448` | What causes and early symptoms indicate hip-joint disease (coxalgia), and why is bed confinement with weight and pulley condemned? | [x] Verified |
| `q_multi_dev_11` | `dev` | `multi_page` | `p0501-p0600` | `p0532, p0533` | What physiological mechanism causes asthmatic paroxysms, and what emergency palliative treatments relieve nocturnal respiratory distress? | [x] Verified |
| `q_multi_dev_12` | `dev` | `multi_page` | `p0501-p0600` | `p0591, p0592` | How are pin-worms (seat-worms) detected in children, and why are localized rectal enemas required instead of oral anthelmintics? | [x] Verified |
| `q_multi_dev_13` | `dev` | `multi_page` | `p0601-p0700` | `p0645, p0646` | How do the clinical symptoms and underlying physiological causes of nervous headache differ from bilious headache? | [x] Verified |
| `q_multi_dev_14` | `dev` | `multi_page` | `p0601-p0700` | `p0684, p0685` | What pathological symptoms characterize acute ear inflammation and chronic purulent otorrhea, and how is the tympanum examined? | [x] Verified |
| `q_multi_dev_15` | `dev` | `multi_page` | `p0701-p0800` | `p0729, p0730` | How does the clinical presentation of neuralgic dysmenorrhea in sensitive temperaments contrast with congestive dysmenorrhea in plethoric patients? | [x] Verified |
| `q_multi_dev_16` | `dev` | `multi_page` | `p0701-p0800` | `p0791, p0792` | How are urinary precipitates of urates and earthy phosphates chemically verified, and why is urinary tract pressure mistaken for Bright's disease? | [x] Verified |
| `q_multi_dev_17` | `dev` | `multi_page` | `p0801-p0900` | `p0841, p0842` | How does spermatorrhea manifest during ordinary daytime physical exertion, and how does reflex nervous action explain its systemic damage? | [x] Verified |
| `q_multi_dev_18` | `dev` | `multi_page` | `p0801-p0900` | `p0882, p0883` | What physical diagnostic tests identify dropsy of the scrotum (hydrocele), and how does suction with aspirator needles safely evacuate fluid? | [x] Verified |
| `q_multi_dev_19` | `dev` | `multi_page` | `p0901-p1034` | `p0906, p0907` | How are abdominal hernias categorized by anatomical location, and what demographic patterns govern their occurrence across age groups and sexes? | [x] Verified |
| `q_multi_dev_20` | `dev` | `multi_page` | `p0901-p1034` | `p0941, p0942` | How is the Invalids' Hotel in Buffalo organized as a specialized chronic disease sanitarium, and what architectural features facilitate patient comfort? | [x] Verified |
| `q_neg_01` | `dev` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | What dosage of oral amoxicillin or intramuscular penicillin is recommended for treating severe bacterial pneumonia? | [x] Verified |
| `q_neg_02` | `dev` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | How does diffusion-weighted magnetic resonance imaging (MRI) detect acute cerebral ischemic stroke within 3 hours of symptom onset? | [x] Verified |
| `q_neg_03` | `dev` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | What mRNA lipid nanoparticle vaccine formulation is administered for immunization against SARS-CoV-2 (COVID-19)? | [x] Verified |
| `q_neg_04` | `dev` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | How is excimer laser photo-refractive keratectomy or LASIK surgery performed to reshape the corneal stroma for myopia correction? | [x] Verified |
| `q_neg_05` | `dev` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | How does CRISPR-Cas9 single guide RNA (sgRNA) direct targeted double-strand DNA cleavage for genome editing in human somatic cells? | [x] Verified |
| `q_test_01` | `test` | `single_page` | `p0001-p0100` | `p0063` | What causes the distinction in color between arterial and venous blood, and what role do the lungs play in this change? | [x] Verified |
| `q_test_02` | `test` | `single_page` | `p0001-p0100` | `p0078` | What are the two primary physiological offices of perspiration in the human body? | [x] Verified |
| `q_test_03` | `test` | `single_page` | `p0101-p0200` | `p0115` | What is the primary function of the crystalline lens in human vision according to Dalton's physiology? | [x] Verified |
| `q_test_04` | `test` | `single_page` | `p0101-p0200` | `p0121` | Which anatomical parts of the human body possess the most acute tactile sensibility according to 19th-century physiology? | [x] Verified |
| `q_test_05` | `test` | `single_page` | `p0201-p0300` | `p0261` | What fourfold purpose do tea and coffee serve when properly employed as beverages? | [x] Verified |
| `q_test_06` | `test` | `single_page` | `p0201-p0300` | `p0273` | Why is flannel considered the best material to wear next to the skin for preserving bodily temperature? | [x] Verified |
| `q_test_07` | `test` | `single_page` | `p0301-p0400` | `p0366` | What is the procedure and physiological effect of the Russian bath in 19th-century hydrotherapy? | [x] Verified |
| `q_test_08` | `test` | `single_page` | `p0401-p0500` | `p0470` | What constitutional taints or physical causes are identified as producing osteitis (inflammation of the bones)? | [x] Verified |
| `q_test_09` | `test` | `single_page` | `p0501-p0600` | `p0590` | What are the two species of taenia (tapeworms) developed in the human intestine, and where is taenia solium commonly found? | [x] Verified |
| `q_test_10` | `test` | `single_page` | `p0601-p0700` | `p0671` | What does the term cataract signify when applied to diseases of the eye? | [x] Verified |
| `q_test_11` | `test` | `single_page` | `p0701-p0800` | `p0798` | What are the two essentially different varieties of diabetes described by medical authorities? | [x] Verified |
| `q_test_12` | `test` | `single_page` | `p0801-p0900` | `p0876` | What anatomical condition constitutes a varicocele, and what does it feel like upon physical examination? | [x] Verified |
| `q_test_13` | `test` | `single_page` | `p0801-p0900` | `p0881` | What pathological accumulation of fluid defines dropsy of the scrotum (hydrocele)? | [x] Verified |
| `q_test_14` | `test` | `single_page` | `p0901-p1034` | `p0921` | What domestic household substances and drinks are recommended as antidotes for iodine poisoning? | [x] Verified |
| `q_test_15` | `test` | `single_page` | `p0901-p1034` | `p0980` | Which organ systems and chronic diseases can be diagnosed with aid from microscopical examination and chemical analysis of the urine? | [x] Verified |
| `q_multi_test_01` | `test` | `multi_page` | `p0301-p0400` | `p0300, p0301` | How did 19th-century rational medicine advance from empiricism, and what are the duties of a scientific practitioner? | [x] Verified |
| `q_multi_test_02` | `test` | `multi_page` | `p0301-p0400` | `p0364, p0365, p0366` | What are the therapeutic purposes and physiological effects of hydrotherapy and medicinal baths according to Dr. Pierce? | [x] Verified |
| `q_multi_test_03` | `test` | `multi_page` | `p0601-p0700` | `p0621, p0622` | What modern medical understanding regarding blood circulation replaces the old practice of blood-letting in apoplexy? | [x] Verified |
| `q_multi_test_04` | `test` | `multi_page` | `p0701-p0800` | `p0789, p0790` | What traumatic, mechanical, and medicinal factors are identified as causes of acute inflammation of the kidneys (acute nephritis)? | [x] Verified |
| `q_multi_test_05` | `test` | `multi_page` | `p0901-p1034` | `p0981, p0980` | How does the invention and application of the microscope assist in detecting chronic diseases from urinary samples at the Invalids Hotel? | [x] Verified |
| `q_neg_06` | `test` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | What is the mechanism of action of HMG-CoA reductase inhibitors (statins) in lowering serum LDL cholesterol in coronary artery disease? | [x] Verified |
| `q_neg_07` | `test` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | How do immune checkpoint inhibitor monoclonal antibodies targeting programmed death receptor-1 (PD-1) enhance anti-tumor T-cell responses? | [x] Verified |
| `q_neg_08` | `test` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | How does a quantitative reverse-transcription polymerase chain reaction (RT-qPCR) assay amplify cDNA to measure viral load? | [x] Verified |
| `q_neg_09` | `test` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | What are the technical advantages of multi-arm robotic laparoscopic platforms (da Vinci Surgical System) in minimally invasive pelvic surgery? | [x] Verified |
| `q_neg_10` | `test` | `out_of_corpus` | `out_of_corpus` | `None (Negative)` | How do quantum computing algorithms and quantum superposition speed up molecular dynamics simulations for drug discovery? | [x] Verified |
