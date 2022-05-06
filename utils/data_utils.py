from data.dataset import INT_TO_LABEL

def ratio_to_category(collection):
    plot_array = []
    nOfImages  = len(collection)
    nOfLabels = 7
    labels = list(INT_TO_LABEL.keys())
    print(labels)
    for i in range(nOfLabels):
        plot_array.append({'id': INT_TO_LABEL[labels[i]]})
        plot_array[i]['data'] = []
        for j,item in enumerate(collection):
            plot_array[i]['data'].append({'x':j, 'y':item['ratio_dict'][i]['value']})
    return plot_array

def get_images(collection):
    imgs = []
    for item in collection:
        imgs.append(item['img'])
    return imgs

#{id: 'Background', data: [{ x: 1997, y:1 },{ x: 28:14, y:1 },{ x: 28:14, y:1 },{ x: 28:14, y:1 }]}'''
#[imgs:[img1, img2,img3], plot_array:[{...}, {...}, {...}, {...}, {...}, {...}, {...}]]