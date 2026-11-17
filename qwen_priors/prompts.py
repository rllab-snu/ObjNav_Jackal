"""All object-prior scoring prompts in one place, grouped by the experiment version they came from.
Kept together so the different 'versions' tried are easy to compare and reuse."""

# --- v1 batched JSON (predict_object_priors, --no-cot): 25 objects in one call. Went lazy. ---
BATCHED = ("You predict where objects would be in an EMPTY home (before anything is placed), for a "
           "robot. Given ONE spot's visible furniture + room type (and maybe an image of that view), "
           "rate 0-1 how likely EACH listed object would end up HERE, from furniture/room affinity "
           "ONLY (mug->kitchen counter/sink; pillow->bed; remote->sofa/tv). Output ONLY a compact "
           "JSON object mapping every item to its score, e.g. {\"mug\": 0.8, ...}.")

# --- v2 per-object CoT (predict_object_priors, --cot): reason then SCORE. Fixed dryer, but saturates. ---
COT = ("You place objects in a home. Look at the photo(s) of ONE spot. In ONE short sentence say "
       "what furniture/surface is visible and whether it is where the given object is usually kept, "
       "then output 'SCORE: <0-1>' (the object's usual furniture clearly visible -> 0.8-1.0; only "
       "wall/curtain/floor/unrelated furniture -> 0.0-0.1).")

# --- v4 per-node anchored ranking (rank_object_priors): 12 dirs in one call, score per reasoning line. ---
RANK = (
    "A robot searches for ONE target object in a home that is EMPTY of small objects now; judge "
    "where the object would normally be KEPT from each viewpoint, using BOTH the photo AND the listed "
    "furniture (the furniture list comes from a reliable scan -- trust it even if the small photo is "
    "unclear). You see several viewpoints of the SAME home. For EACH spot output a line "
    "'spot i: <what/which furniture> = <score>':\n"
    "  0.8-1.0  the target's TYPICAL storage surface is present (hair dryer->dressing table/vanity; "
    "mug->kitchen counter/sink/dining table; remote->tv unit/sofa/coffee table; pillow->bed).\n"
    "  0.3-0.5  furniture is present but NOT the target's typical home (e.g. a bed for a hair dryer).\n"
    "  0.0-0.1  only wall, curtain, window, floor, doorway or empty space -- nothing can rest there.\n"
    "Score each spot on its own surface. Then repeat all as one JSON line {id: score}.")

# --- v5 sharpened per-object, VLM (gen_direction --signal vlm): specific furniture->high, generic->0.3, wall->0. ---
SHARP_VLM = (
    "You place ONE object in a home; judge if THIS view is where it is usually kept, from the "
    "furniture/surface actually visible (use BOTH the photos and the listed furniture; trust the "
    "furniture list even if the photo is unclear). First name the main surface you see in <=6 words, "
    "then output 'SCORE: <0-1>'. Scoring: 0.8-1.0 ONLY if the object's SPECIFIC typical furniture is "
    "present (hair dryer->dressing table/vanity; mug->kitchen counter/sink/dining table; remote->tv "
    "unit/sofa/coffee table; pillow->bed); 0.3-0.5 if a generic surface that is NOT this object's "
    "usual home (a bare side table, a bed for a dryer, a chair); 0.0 if only wall, curtain, window, "
    "floor, doorway or empty space. Do NOT infer high from the room alone.")

# --- v6 sharpened per-object, LLM from caption (gen_direction --signal caption): same tiers, text only. ---
SHARP_LLM = (
    "You judge if a spot is where ONE object is usually kept, from a short description of what is "
    "visible there plus the room. Output ONLY 'SCORE: <0-1>'. 0.8-1.0 if the object's SPECIFIC "
    "usual furniture/surface is described (hair dryer->dressing table/vanity; mug->kitchen "
    "counter/sink/dining table; remote->tv unit/sofa/coffee table; pillow->bed); 0.3-0.5 if a "
    "generic surface that is not its usual home; 0.0 if only wall/curtain/window/floor/door/empty "
    "is described (nothing can rest there).")

# --- room-level affinity signal (LLM; hard gate in two_stage, soft prior in current ensemble) ---
ROOM = ("You judge how likely a home keeps ONE object in a given ROOM, from the room type and the "
        "furniture there. Output ONLY 'SCORE: <0-1>' for room-level affinity (mug->kitchen; "
        "pillow/hair dryer->bedroom; remote->living room).")

# --- caption (annotate_captions): describe visible surfaces, or say only wall/curtain/empty. ---
CAPTION = ("Describe ONLY what a robot camera sees in this view, in ONE short sentence, for deciding "
           "where objects could be placed. Name the visible furniture/surfaces (sofa, small coffee "
           "table, kitchen counter, dresser, bed, tv unit...). If only a bare wall, curtain, window, "
           "door or empty floor is visible with no usable surface, say exactly that. Do not guess "
           "objects that are not visible.")

# --- node montage scoring (qwen_score.score_node, used by bench matrix) ---
NODE = ("You estimate where a target object is in a home, for a robot. Given a location's "
        "visible furniture / previously-seen objects (and maybe an image), rate how likely the "
        "TARGET is at that location. Homes are dynamic: past objects are a hint, not a guarantee; "
        "reason from furniture/room context (mug -> kitchen counter/sink; pillow -> bed). "
        "Reply EXACTLY one line: 'SCORE: <0-1> | REASON: <short clause>'.")
