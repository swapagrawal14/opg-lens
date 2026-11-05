// Explanation cards. DRAFT: to be reviewed and approved by Dr. Swapnil Agrawal before public launch.
const FINDINGS = {
  "Caries": {
    color: "#facc15", hi: "दाँत में सड़न (कैविटी)",
    looks: "A dark (radiolucent) area in the crown enamel/dentin that does not reach the pulp. Often on proximal surfaces near the contact point or under old restorations.",
    why: "Acid from plaque bacteria dissolves the mineral, so that spot lets more X-rays through and looks darker.",
    mimics: "Cervical burnout (dark band at the neck between enamel and bone), Mach band effect at the DEJ, overlapping contacts on OPG.",
    next: "Confirm with a clinical exam and a bitewing. Small lesions may be managed with fluoride/sealing; cavitated lesions usually need a restoration.",
    patient: "an early cavity (tooth decay) that has not reached the nerve",
    patientHi: "शुरुआती सड़न (कैविटी) जो अभी नस तक नहीं पहुँची है"
  },
  "Deep Caries": {
    color: "#fb923c", hi: "गहरी सड़न",
    looks: "A large radiolucency through the dentin that reaches or nearly reaches the pulp chamber.",
    why: "Decay has progressed close to the pulp, so the radiolucent zone merges with or lies very near the pulp outline.",
    mimics: "Superimposition on OPG can make lesions look closer to the pulp than they are; check with an IOPA.",
    next: "Check pulp vitality and symptoms; an IOPA helps decide between indirect pulp capping, pulpotomy, or root canal treatment.",
    patient: "deep tooth decay that is close to the nerve",
    patientHi: "गहरी सड़न जो नस के बहुत पास है"
  },
  "Periapical Lesion": {
    color: "#f472b6", hi: "जड़ के सिरे पर संक्रमण / घाव",
    looks: "A radiolucency around the root apex with loss of the lamina dura, sometimes with a corticated border.",
    why: "Infection from a necrotic pulp spreads out of the root tip and resorbs the surrounding bone (granuloma, cyst or abscess).",
    mimics: "Mental foramen near lower premolars, incisive canal, maxillary sinus over upper molar roots, normal PDL space.",
    next: "Correlate with vitality tests and an IOPA (or CBCT if unclear). Usually managed with root canal treatment or extraction.",
    patient: "a sign of infection at the tip of the tooth root",
    patientHi: "दाँत की जड़ के सिरे पर संक्रमण का संकेत"
  },
  "Impacted": {
    color: "#38bdf8", hi: "फँसा हुआ दाँत",
    looks: "A tooth that has not erupted into its normal position, often a third molar that is angled (mesioangular, horizontal, vertical, distoangular) against the tooth in front or in bone.",
    why: "Lack of space, an abnormal path of eruption, or an obstruction stops the tooth from coming in.",
    mimics: "A normally developing unerupted tooth in a young patient; check the patient's age and root development.",
    next: "Assess angulation (Winter's), depth (Pell and Gregory), and relation to the inferior alveolar canal. Symptomatic or risky impactions are usually removed.",
    patient: "a tooth that is stuck and has not come in properly (often a wisdom tooth)",
    patientHi: "एक दाँत जो फँसा हुआ है और ठीक से बाहर नहीं आया (अक्सर अक्ल दाढ़)"
  }
};
const DIS = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"];
const FDI = []; for (let q = 1; q <= 4; q++) for (let t = 1; t <= 8; t++) FDI.push(`${q}${t}`);
const TOOTH = { 1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar", 5: "second premolar", 6: "first molar", 7: "second molar", 8: "third molar (wisdom tooth)" };
const QUAD = { 1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right" };
const QUAD_HI = { 1: "ऊपर दाएँ", 2: "ऊपर बाएँ", 3: "नीचे बाएँ", 4: "नीचे दाएँ" };
function toothName(fdi) { if (!fdi || fdi === "?") return "tooth not numbered"; return `${QUAD[fdi[0]]} ${TOOTH[fdi[1]]}`; }
