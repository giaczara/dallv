import torch
import src.models.components.clip as clip

from json import load
from random import sample


def text_prompt(classes_names, dataset=None, n_templates=-1, image_templates="none"):
    text_aug = [
        f"a photo of action {{}}",
        f"a picture of action {{}}",
        f"Human action of {{}}",
        f"{{}}, an action",
        f"{{}} this is an action",
        f"{{}}, a video of action",
        f"Playing action of {{}}",
        f"{{}}",
        f"Playing a kind of action, {{}}",
        f"Doing a kind of action, {{}}",
        f"Look, the human is {{}}",
        f"Can you recognize the action of {{}}?",
        f"Video classification of {{}}",
        f"A video of {{}}",
        f"The man is {{}}",
        f"The woman is {{}}",
    ]
    if image_templates == "simple":
        text_aug = [
            f"a photo of action {{}}",
            f"a picture of action {{}}",
            f"Human action of {{}}",
            f"{{}}, an action",
            f"{{}} this is an action",
            f"{{}}, a photo of action",
            f"Playing action of {{}}",
            f"{{}}",
            f"Playing a kind of action, {{}}",
            f"Doing a kind of action, {{}}",
            f"Look, the human is {{}}",
            f"Can you recognize the action of {{}}?",
            f"Image classification of {{}}",
            f"An image of {{}}",
            f"The man is {{}}",
            f"The woman is {{}}",
        ]
    elif image_templates == "clip":
        text_aug = [
            "a bad photo of a {}",
            "a photo of many {}",
            "a sculpture of a {}",
            "a photo of the hard to see {}",
            "a low resolution photo of the {}",
            "a rendering of a {}",
            "graffiti of a {}",
            "a bad photo of the {}",
            "a cropped photo of the {}",
            "a tattoo of a {}",
            "the embroidered {}",
            "a photo of a hard to see {}",
            "a bright photo of a {}",
            "a photo of a clean {}",
            "a photo of a dirty {}",
            "a dark photo of the {}",
            "a drawing of a {}",
            "a photo of my {}",
            "the plastic {}",
            "a photo of the cool {}",
            "a close-up photo of a {}",
            "a black and white photo of the {}",
            "a painting of the {}",
            "a painting of a {}",
            "a pixelated photo of the {}",
            "a sculpture of the {}",
            "a bright photo of the {}",
            "a cropped photo of a {}",
            "a plastic {}",
            "a photo of the dirty {}",
            "a jpeg corrupted photo of a {}",
            "a blurry photo of the {}",
            "a photo of the {}",
            "a good photo of the {}",
            "a rendering of the {}",
            "a {} in a video game",
            "a photo of one {}",
            "a doodle of a {}",
            "a close-up photo of the {}",
            "a photo of a {}",
            "the origami {}",
            "the {} in a video game",
            "a sketch of a {}",
            "a doodle of the {}",
            "a origami {}",
            "a low resolution photo of a {}",
            "the toy {}",
            "a rendition of the {}",
            "a photo of the clean {}",
            "a photo of a large {}",
            "a rendition of a {}",
            "a photo of a nice {}",
            "a photo of a weird {}",
            "a blurry photo of a {}",
            "a cartoon {}",
            "art of a {}",
            "a sketch of the {}",
            "a embroidered {}",
            "a pixelated photo of a {}",
            "itap of the {}",
            "a jpeg corrupted photo of the {}",
            "a good photo of a {}",
            "a plushie {}",
            "a photo of the nice {}",
            "a photo of the small {}",
            "a photo of the weird {}",
            "the cartoon {}",
            "art of the {}",
            "a drawing of the {}",
            "a photo of the large {}",
            "a black and white photo of a {}",
            "the plushie {}",
            "a dark photo of a {}",
            "itap of a {}",
            "graffiti of the {}",
            "a toy {}",
            "itap of my {}",
            "a photo of a cool {}",
            "a photo of a small {}",
            "a tattoo of the {}"
        ]

    if n_templates == 1:
        text_aug = [f"A video of {{}}"]
    elif n_templates > 1:
        text_aug = [f"A video of {{}}"] + sample(text_aug, n_templates - 1)

    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat(
            [clip.tokenize(txt.format(c)) for i, c in classes_names]
        )

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


def manually_enriched_text_prompt(classes_names, dataset):

    if "ek" in dataset:
        class_map = {
            "taking": "a person taking an object with his hands",
            "putting": "a person placing an object somewhere",
            "closing": "a person closing a door or a lid with his hands",
            "opening": "a person opening a door or a lid with his hands",
            "washing": "a person washing an object with water and soap",
            "pouring": "a person transferring some liquid into a container",
            "mixing": "a person mixing together two or more ingredients",
            "cutting": "a person using a blade to cut an object",
        }
    elif "hmdb" in dataset:
        class_map = {
            "climb": "a person climbing a wall on the rocks or in a gym",
            "fencing": "a person practicing the sport of fencing with swords",
            "golf": "a person playing golf on the grass with a club",
            "kick ball": "a person kicking a ball",
            "pullup": "a person doing workout with pull-ups",
            "punch": "a person punching another person or a punching bag",
        }
    else:
        raise ValueError("Class map not available for the {} dataset!".format(dataset))

    text_dict = {0: torch.cat([clip.tokenize(class_map[c]) for _, c in classes_names])}

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, 1, text_dict


def gpt_text_prompt(classes_names, dataset):

    if "ek" in dataset:
        class_map = {
            "taking": "grasping or holding with the hand",
            "putting": "placing or setting something in a particular location",
            "closing": "shutting or making something inaccessible or unavailable by covering or obstructing it",
            "opening": "making something accessible or available by moving or removing a covering or obstruction",
            "washing": "cleaning something using water and often soap or another cleaning agent.",
            "pouring": "causing a liquid to flow from a container by tilting it or by turning it upside down",
            "mixing": "combining or blending together two or more things",
            "cutting": "dividing or separating something into pieces using a sharp tool or object",
        }
    elif dataset in ["hmdb_ucf", "ucf_hmdb"]:
        class_map = {
            "climb": "ascending a vertical or steep surface using your hands, feet, or other body parts to hold onto or pull yourself up. It can be done for recreation, exercise, or as a means of reaching a specific destination",
            "fencing": "a sport that involves two opponents trying to score points by using swords to touch or hit each other",
            "golf": "a sport in which players use clubs to hit a small ball into a series of holes on a course",
            "kick ball": "using your foot to strike or propel a ball through the air",
            "pullup": "a type of exercise that involves pulling your body up towards a stationary bar using your upper body strength",
            "punch": "the act of striking someone or something with a closed fist",
        }
    elif dataset in [
        "kinetics_hmdb",
        "kinetics_arid",
        "hmdb_kinetics",
        "arid_kinetics",
    ]:
        class_map = {
            "drink": "consuming a liquid, usually water or a beverage, through the mouth",
            "jump": "propelling oneself upward or forward into the air using the legs and muscles",
            "pick": "selecting or choosing something, often with the fingers or a tool like a pick",
            "pour": "causing a liquid or other flowable substance to flow out of a container by tilting or overturning it",
            "push": "applying force to an object in order to move it away from oneself or in the opposite direction of the force being applied",
            "run": "moving quickly on foot by taking steps in which both feet are off the ground at the same time",
            "walk": "moving on foot by taking steps in which one foot is on the ground at a time",
            "wave": "moving your hand or arm back and forth or up and down",
        }
    else:
        raise ValueError("Class map not available for the {} dataset!".format(dataset))

    for _, c in classes_names:
        assert c in class_map

    text_dict = {0: torch.cat([clip.tokenize(class_map[c]) for _, c in classes_names])}

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, 1, text_dict


def merged_text_prompt(classes_names, dataset):
    if "ek" in dataset:
        class_map = {
            "taking": [
                "grasping or holding with the hand",
                "a person taking an object with his hands",
            ],
            "putting": [
                "placing or setting something in a particular location",
                "a person placing an object somewhere",
            ],
            "closing": [
                "shutting or making something inaccessible or unavailable by covering or obstructing it",
                "a person closing a door or a lid with his hands",
            ],
            "opening": [
                "making something accessible or available by moving or removing a covering or obstruction",
                "opening a door or a lid with his hands",
            ],
            "washing": [
                "cleaning something using water and often soap or another cleaning agent",
                "washing an object with water and soap",
            ],
            "pouring": [
                "causing a liquid to flow from a container by tilting it or by turning it upside down",
                "transferring some liquid into a container",
            ],
            "mixing": [
                "combining or blending together two or more things",
                "mixing together two or more ingredients",
            ],
            "cutting": [
                "dividing or separating something into pieces using a sharp tool or object",
                "using a blade to cut an object",
            ],
        }

        text_dict = {}
        num_text_aug = 2

        for i in range(num_text_aug):
            text_dict[i] = torch.cat(
                [clip.tokenize(class_map[c][i]) for _, c in classes_names]
            )

        classes = torch.cat([v for k, v in text_dict.items()])

        return classes, num_text_aug, text_dict


def hierarchical_text_prompt(
    classes_names,
    dataset,
    version=1,
):
    templates = [
        f"a photo of action {{}}",
        f"a picture of action {{}}",
        f"Human action of {{}}",
        f"{{}}, an action",
        f"{{}} this is an action",
        f"{{}}, a video of action",
        f"Playing action of {{}}",
        f"{{}}",
        f"Playing a kind of action, {{}}",
        f"Doing a kind of action, {{}}",
        f"Look, the human is {{}}",
        f"Can you recognize the action of {{}}?",
        f"Video classification of {{}}",
        f"A video of {{}}",
        f"The man is {{}}",
        f"The woman is {{}}",
    ]

    json_file = "data/hierarchical_prompts.json"
    with open(json_file, "r") as json_file:
        class_maps = load(json_file)

    if dataset in ["hmdb_ucf", "ucf_hmdb"]:
        class_map = class_maps["hmdb_ucf"]["v{}".format(version)]
    elif dataset in [
        "kinetics_hmdb",
        "kinetics_arid",
        "hmdb_kinetics",
        "arid_kinetics",
        "hmdb_arid",
        "arid_hmdb",
        "hmdb_mit",
        "mit_hmdb",
        "kinetics_mit",
        "arid_mit",
        "mit_kinetics",
        "mit_arid",
    ]:
        class_map = class_maps["daily_da"]["v{}".format(version)]
    elif "ek" in dataset:
        class_map = class_maps["epic_kitchens"]["v{}".format(version)]
    elif dataset == "hmdb51":
        class_map = class_maps["hmdb51"]
    else:
        raise ValueError("Hierarchical prompts for {} not available!".format(dataset))

    num_text_aug = len(class_map[list(class_map.keys())[0]])
    for _, c in classes_names:
        assert c in class_map, "Class {} not in class map!".format(c)
        assert len(class_map[c]) == num_text_aug

    classes_by_template = []
    texts = []
    for i in range(num_text_aug):
        texts.extend([class_map[c][i] for _, c in classes_names])

    for t in templates:
        text_dict = {}
        for i in range(num_text_aug):
            text_dict[i] = torch.cat(
                [clip.tokenize(t.format(class_map[c][i])) for _, c in classes_names]
            )

        classes = torch.cat([v for k, v in text_dict.items()])
        classes_by_template.append(classes)

    classes = torch.stack(classes_by_template)

    return classes, num_text_aug, text_dict, len(templates), texts
