from data.dataset import INT_TO_LABEL, LABEL_TO_HSL

def ratio_to_category(collection):
    plot_array = []
    nOfImages  = len(collection)
    nOfLabels = 7
    labels = list(INT_TO_LABEL.values())
    print(labels)
    for i in range(nOfLabels):
        plot_array.append({'id': labels[i]})
        plot_array[i]['color'] = LABEL_TO_HSL[labels[i]]
        plot_array[i]['data'] = []
        for j,item in enumerate(collection):
            plot_array[i]['data'].append({'x':j, 'y':item['ratio_dict'][i]['value']})
    return plot_array

def get_images(collection):
    imgs = []
    for item in collection:
        imgs.append(item['img'])
    return imgs