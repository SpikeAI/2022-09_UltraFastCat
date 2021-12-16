
#import model's script and set the output file

#from UltraFastCat.model import *
from experiment_train import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

filename = f'results/{datetag}_results_3_{args.HOST}.json'
print(f'{filename=}')

def main():
    if os.path.isfile(filename):
        df_gray = pd.read_json(filename)
    else:
        i_trial = 0
        df_gray = pd.DataFrame([], columns=['model', 'model_task', 'task', 'goal', 'likelihood', 'fps', 'time', 'i_image', 'filename', 'top_1', 'device_type']) 
            # image preprocessing
        (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=args.image_size, batch_size=1, p=1)
        for task in args.tasks:
            pprint(task)
            for i_image, (data, label) in enumerate(dataloaders[task]['test']):
                data, label = data.to(device), label.to(device)
                for model_name in all_models:
                    model = models_vgg[model_name].to(device)
                    with torch.no_grad():
                        goal = 'target' if 'target' in image_datasets[task]['test'].imgs[i_image][0] else 'distractor'
                        model_task = 'animal' if 'animal' in model_name else 'artifact'
                        tic = time.time()
                        out = model(data).squeeze(0)
                        percentage = float((torch.sigmoid(out) * 100).detach().cpu().numpy()[0])
                        top_1 = 'target' if percentage>50 else 'distractor'
                        elapsed_time = time.time() - tic
                    df_gray.loc[i_trial] = {'model':model_name,'model_task':model_task, 'task':task, 'top_1':top_1, 'goal':goal, 'likelihood':percentage, 'time':elapsed_time, 'fps': 1/elapsed_time,
                                          'i_image':i_image, 'filename':image_datasets[task]['test'].imgs[i_image][0], 'device_type':device.type}
                    print(f'The {model_name} model categorize {model_task} with {percentage:.3f} % likelihood ({top_1}) in {elapsed_time:.3f} seconds, groundtrue : {task}, {goal}')
                    i_trial += 1
        df_gray.to_json(filename)
main()     
