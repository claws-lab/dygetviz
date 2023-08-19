color_to_node_type = {
    "#FF0000": "highlighted",
    "#FFBF00": "reference",
    "#00FF11": "projected",
    "#B2B2B2": "background",
    "#EF476F": "anomaly",

}

node_type_to_size = {
    "highlighted": 60,
    "reference": 20,
    "background": 5,
    "anomaly": 5,
}

node_type_to_color = {v: k for k, v in color_to_node_type.items()}

pure_color_palettes = ["Blues", "Reds", "Yellows", "Greens", "Oranges", "Purples", "Greys"]
