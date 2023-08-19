from const_viz import *

def adjust_node_color_size(fig):
    for i in range(len(fig.data)):
        color = fig.data[i]['name']
        node_type = color_to_node_type[color]
        fig.data[i]['legendgroup'] = fig.data[i]['name'] = node_type
        fig.data[i]['marker']['color'] = color
        fig.data[i]['marker']['size'] = node_type_to_size[node_type]
    return fig