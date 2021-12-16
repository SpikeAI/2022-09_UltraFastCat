
from src.init import *  
verbose = False

with open(args.url_loader) as json_file:
    Imagenet_urls_ILSVRC_2016 = json.load(json_file)

def clean_list(list_dir, patterns=['.DS_Store']):
    for pattern in patterns:
        if pattern in list_dir: list_dir.remove('.DS_Store')
    return list_dir

import imageio
def get_image(img_url, timeout=3., min_content=3, verbose=verbose):
    try:
        img_resp = imageio.imread(img_url)
        if (len(img_resp.shape) < min_content): # TODO : raise error when min_content is not reached
            if verbose : print(f"Url {img_url} does not have enough content")
            return False
        else:
            if verbose : print(f"Success with url {img_url}")
            return img_resp
    except Exception as e:
        if verbose : print(f"Failed with {e} for url {img_url}")
        return False # did not work

import hashlib # jah.

# train, val and test folders
    
list_urls = {}
list_img_name_used = {}
for task in class_wnids :
    list_urls[task] = {}
    list_img_name_used[task] = {}
    for goal in class_wnids[task]:
        list_urls[task][goal] = {}
        for class_wnid in class_wnids[task][goal]:
            list_urls[task][goal][class_wnid] = Imagenet_urls_ILSVRC_2016[str(class_wnid)]
        np.random.shuffle(list_urls[task][goal][class_wnid])
        list_img_name_used[task][goal] = []
            
    
# train, val and test folders
for task in args.tasks:
    pprint(f'Task \"{task}\"')
    for folder in args.folders :
        print(f'Folder \"{folder}\"')
        filename = f'results/{datetag}_dataset_{folder}_{args.HOST}.json'
        columns = ['task', 'goal', 'img_url', 'img_name', 'is_flickr', 'dt', 'worked', 'class_wnid', 'class_name']
        if os.path.isfile(filename):
            df_dataset = pd.read_json(filename)
        else:
            df_dataset = pd.DataFrame([], columns=columns)
        for goal in args.goals :
            print(f'Scraping images for task \"{task}\" : {goal} ')
            task_goal_folder = os.path.join(paths[task][folder], goal)
            os.makedirs(task_goal_folder, exist_ok=True)
            list_img_name_used[task][goal] += clean_list(os.listdir(os.path.join(paths[task][folder], goal))) # join two lists
            while (len(clean_list(os.listdir(task_goal_folder))) < N_images_per_class[folder]):
                class_wnid = np.random.choice(class_wnids[task][goal], 1)[0]
                pos = class_wnids[task][goal].index(class_wnid)
                class_name = reverse_id_labels[class_wnid]
                # pick and remove element from shuffled list
                if len(list_urls[task][goal][class_wnid])==0:
                    del class_wnids[task][goal][pos], list_urls[task][goal][class_wnid]
                else:
                    img_url = list_urls[task][goal][class_wnid].pop()

                    if len(df_dataset[df_dataset['img_url']==img_url])==0 : # we have not yet tested this URL yet

                        # Transform URL into filename
                        # https://laurentperrinet.github.io/sciblog/posts/2018-06-13-generating-an-unique-seed-for-a-given-filename.html
                        img_name = hashlib.sha224(img_url.encode('utf-8')).hexdigest() + '.png'

                        if img_url.split('.')[-1] in ['.tiff', '.bmp', 'jpe', 'gif']:
                            if verbose: print('Bad extension for the img_url', img_url)
                            worked, dt = False, 0.
                        # make sure it was not used in other folders
                        elif not (img_name in list_img_name_used[task][goal]):
                            tic = time.time()
                            img_content = get_image(img_url, verbose=verbose)
                            dt = time.time() - tic
                            worked = img_content is not False
                            if worked:
                                if verbose : print('Good URl, now saving', img_url, ' in', task_goal_folder, ' as', img_name)
                                imageio.imsave(os.path.join(task_goal_folder, img_name), img_content, format='png')
                                list_img_name_used[task][goal].append(img_name)
                        df_dataset.loc[len(df_dataset.index)] = {'task': task, 'goal': goal, 'img_url':img_url, 'img_name':img_name, 'is_flickr':1 if 'flickr' in img_url else 0, 'dt':dt,
                                        'worked':worked, 'class_wnid':class_wnid, 'class_name':class_name}
                        df_dataset.to_json(filename)
                        print(f'\r{len(clean_list(os.listdir(task_goal_folder)))} / {N_images_per_class[folder]}', end='\n' if verbose else '', flush=not verbose)

        if (len(clean_list(os.listdir(task_goal_folder))) < N_images_per_class[folder]) and (len(list_urls[task][goal]) == 0): 
            print('Not enough working url to complete the dataset') 

    df_dataset.to_json(filename)

print('\n')
pprint(f'Some random images :')
import imageio
N_image_i = 4
image_plot_paths = {}
task_folder = {}
x = 0
folder = 'train'
fig, axs = plt.subplots(len(args.tasks)*len(args.goals), N_image_i, figsize=(fig_width, fig_width))
for task in args.tasks:
    for goal in args.goals:
        task_goal_folder = os.path.join(paths[task][folder], goal)
        image_plot_paths = os.listdir(task_goal_folder)
        for i_image in np.arange(N_image_i):
            ax = axs[x][i_image]
            path = os.path.join(task_goal_folder, image_plot_paths[i_image])
            ax.imshow(imageio.imread(path))
            ax.set_xticks([])
            ax.set_yticks([])  
            if i_image%5 == 0:
                ax.set_ylabel((task+' '+goal), fontsize = 18)
        x +=1
fig.set_facecolor(color='white')
