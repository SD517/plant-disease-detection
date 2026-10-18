"""Turn PlantVillage folder names into readable text and short advice.

The advice is general, educational guidance. It is not a substitute for a
local agricultural extension officer or plant pathologist.
"""
from __future__ import annotations

import re


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _pretty(text: str) -> str:
    text = text.replace("_", " ").replace(",", ", ")
    return re.sub(r"\s+", " ", text).strip(" ,")


def parse_class_name(raw: str) -> tuple[str, str, bool]:
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight', False)."""
    if "___" in raw:
        crop, cond = raw.split("___", 1)
    elif "__" in raw:
        crop, cond = raw.split("__", 1)
    else:
        crop, _, cond = raw.partition("_")
    crop, cond = _pretty(crop), _pretty(cond)
    healthy = _norm(cond) == "healthy"
    if healthy:
        cond = "Healthy"
    return crop, cond, healthy


# key (normalised substring of the condition) -> (cause, description, advice)
# Order matters: more specific keys first.
_DISEASES = [
    ("northernleafblight", "Fungal",
     "Long grey-green to tan cigar-shaped lesions on maize leaves, caused by Exserohilum turcicum.",
     ["Plant resistant hybrids where available.", "Rotate crops and bury or remove infected residue.",
      "Ask your extension officer about timely fungicide use if the disease starts early."]),
    ("cercospora", "Fungal",
     "Rectangular grey to tan lesions that run between the leaf veins (grey leaf spot) on maize.",
     ["Rotate away from maize for at least one season.", "Manage crop residue and improve air flow.",
      "Choose tolerant hybrids; fungicide only on local advice."]),
    ("commonrust", "Fungal",
     "Small reddish-brown pustules on both leaf surfaces, caused by Puccinia sorghi.",
     ["Use resistant varieties.", "Scout early; severe cases may need a fungicide on local advice."]),
    ("isariopsis", "Fungal",
     "Dark irregular leaf spots on grape leaves (Isariopsis leaf spot) that can cause early leaf drop.",
     ["Prune for open canopy and good air movement.", "Remove fallen infected leaves.",
      "Follow a local grape spray programme."]),
    ("esca", "Fungal complex",
     "Also called black measles: tiger-stripe leaf discolouration and spotted berries on grapevines.",
     ["Prune in dry weather and protect large pruning wounds.", "Remove and destroy badly affected vines.",
      "Seek advice from a viticulture specialist; there is no simple cure."]),
    ("haunglongbing", "Bacterial (insect-spread)",
     "Citrus greening: blotchy yellow leaves and misshapen bitter fruit, spread by the Asian citrus psyllid.",
     ["Remove infected trees to protect healthy ones.", "Control psyllid insects.",
      "Report suspected cases to the local agriculture authority."]),
    ("spidermites", "Pest",
     "Tiny two-spotted spider mites cause fine yellow stippling and webbing on the underside of leaves.",
     ["Wash plants with a strong water spray.", "Encourage natural predators and avoid dusty, dry conditions.",
      "Use miticides only as a last resort and as locally advised."]),
    ("yellowleafcurl", "Viral (whitefly-spread)",
     "Leaves curl upward, turn yellow and growth is stunted. The virus is carried by whiteflies.",
     ["Control whiteflies and use insect-proof nets for seedlings.", "Remove and destroy infected plants.",
      "Plant tolerant varieties."]),
    ("mosaic", "Viral",
     "Mottled light and dark green patterns and distorted leaves.",
     ["Remove infected plants; the virus spreads by touch and tools.", "Wash hands and disinfect tools.",
      "Use certified virus-free seed and resistant varieties."]),
    ("leafmold", "Fungal",
     "Pale yellow spots on the upper leaf surface with olive-brown mould underneath, common in humid greenhouses.",
     ["Lower humidity and ventilate.", "Avoid wetting leaves when watering.", "Remove heavily infected leaves."]),
    ("septoria", "Fungal",
     "Many small circular spots with dark borders and pale centres, usually starting on lower leaves.",
     ["Remove infected lower leaves.", "Mulch and water at the base to limit splashing.",
      "Rotate crops; fungicide on local advice."]),
    ("targetspot", "Fungal",
     "Brown spots with concentric rings, like a target, caused by Corynespora cassiicola.",
     ["Improve air circulation and avoid overhead watering.", "Remove infected debris.",
      "Rotate crops; fungicide on local advice."]),
    ("leafscorch", "Fungal",
     "Dark purple blotches on strawberry leaves that merge so the leaf looks scorched.",
     ["Remove infected leaves after harvest.", "Avoid overhead irrigation.", "Plant resistant varieties."]),
    ("applescab", "Fungal",
     "Olive-green to brown scabby spots on apple leaves and fruit, caused by Venturia inaequalis.",
     ["Rake and destroy fallen leaves.", "Prune for air flow.", "Use a spring protectant programme as locally advised."]),
    ("cedarapplerust", "Fungal",
     "Bright yellow-orange spots on apple leaves; the fungus needs nearby juniper or cedar to complete its life cycle.",
     ["Remove nearby juniper hosts if practical.", "Plant resistant apple varieties.",
      "Protectant sprays in spring on local advice."]),
    ("powderymildew", "Fungal",
     "White powdery fungal growth on leaf surfaces that can curl and weaken leaves.",
     ["Improve air flow and avoid crowding.", "Remove badly infected leaves.",
      "Treat early with locally approved products."]),
    ("blackrot", "Fungal",
     "Brown leaf spots with darker edges that later bear tiny black fruiting bodies; can also rot fruit.",
     ["Prune out and destroy infected wood and mummified fruit.", "Keep the canopy open and dry.",
      "Follow a local protectant spray schedule."]),
    ("bacterialspot", "Bacterial",
     "Small dark water-soaked spots that turn brown or black, spread by rain splash and contaminated seed.",
     ["Use disease-free seed or transplants.", "Avoid working with wet plants.",
      "Copper-based sprays may help; follow local guidance and rotate crops."]),
    ("earlyblight", "Fungal",
     "Brown spots with concentric rings (target-like) on older leaves, caused by Alternaria species.",
     ["Remove infected lower leaves.", "Rotate crops and mulch.", "Water at the base; fungicide on local advice."]),
    ("lateblight", "Oomycete (water mould)",
     "Large, greasy dark patches that spread fast in cool, wet weather; can destroy a crop in days.",
     ["Act quickly: remove and destroy infected plants.", "Avoid overhead watering and improve drainage.",
      "Contact your extension service urgently about protective fungicides."]),
]

_HEALTHY_TEXT = ("No sign of disease was detected on this leaf.",
                 ["Keep monitoring the plant regularly.", "Water at the base and keep good spacing.",
                  "Re-check if new spots or colour changes appear."])


def get_info(raw: str) -> dict:
    """Readable information for a dataset class name."""
    crop, cond, healthy = parse_class_name(raw)
    if healthy:
        desc, advice = _HEALTHY_TEXT
        return {"raw": raw, "crop": crop, "condition": cond, "healthy": True,
                "cause": "None", "description": desc, "advice": advice}
    key = _norm(cond)
    for k, cause, desc, advice in _DISEASES:
        if k in key:
            return {"raw": raw, "crop": crop, "condition": cond, "healthy": False,
                    "cause": cause, "description": desc, "advice": advice}
    return {"raw": raw, "crop": crop, "condition": cond, "healthy": False,
            "cause": "Unknown", "description": "A disease class from the PlantVillage dataset.",
            "advice": ["Consult a local agricultural extension officer."]}
