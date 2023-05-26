"""
Jeremie-etal-Vision_video-abstract 
----------------------------------

# Creating a video abstract for our paper as a movie using the (*excellent*) [MoviePy](http://zulko.github.io/moviepy/index.html) library:

    * Using the template @ https://laurentperrinet.github.io/sciblog/posts/2019-09-11_video-abstract-vision.html
    * and using that @ https://github.com/chloepasturel/AnticipatorySPEM/blob/master/2020-03_video-abstract/2020-03-24_video-abstract.ipynb
    * or that more recent one: https://github.com/SpikeAI/2022_polychronies-review/blob/main/2022-12-23_video-abstract.py


"""


videoname = 'Jeremie-etal-Vision_video-abstract'
gifname = videoname + ".gif"
gifname = None
fps = 3
from moviepy.editor import VideoFileClip, ImageClip, TextClip, CompositeVideoClip

H, W = 500, 800
H_fig, W_fig = int(H-H/(1.618*3)), int(W-W/(1.618*3))


opt_t = dict(font="Arial", size=(W,H), method='caption')
opt_st = dict(font="Arial", size=(W,H), method='caption')
# opt_t = dict(size=(W,H), method='caption')
# opt_st = dict(size=(W,H), method='caption')

clip = []
t = 0 

#################################################################################
# TITRE
#################################################################################
# 
texts = [
    """
    Ultrafast Image Categorization
     in Biology and Neural Models

    by JN Jérémie and L Perrinet

    *MDPI Vision* (2023)"""]
txt_opts = dict(align='center', color='white', **opt_t) #stroke_color='gray', stroke_width=.5
duration = 3

for text in texts:
    txt = TextClip(text, fontsize=35, **txt_opts).set_start(t).set_duration(duration)
    t += duration
    clip.append(txt)

#################################################################################
# INTRO
#################################################################################


#################################################################################
#################################################################################
chapters = [dict(title="Ultrafast visual categorisation", color='green',
            content=[dict(figure='figures_video/find-cheetah.jpg', duration=11, subtitle=[
                            "What distinguishes a visual scene that includes an animal ...", 
                            "from one that does not?",
                            "This question of “animacy detection” is crucial ...", 
                            "for the survival of any species,",
                            "especially in regard to the interactions ...",
                            "between prey and predators.",
                            "Yet human categorization of an animal can be performed ...", 
                            ]),
                     dict(figure='figures_video/found_cheetah.png', duration=3, subtitle=[
                            "very quickly and with high accuracy."])]), 
           dict(title="CNNs", color='orange',
            content=[dict(figure='figures_video/vgg_16_transfer_learning_architecture.png', duration=5, subtitle=[
                           "We choose to use the VGG16 CNN architecture ...",
                           "on a ecological task such as finding an animal ...",
                           "on a custom variant of the Imagenet Challenge dataset ..."]),
                    dict(figure='figures_video/fig_wordnet.png', duration=7, subtitle=[
                           "In particular, we show that by exploiting the  ...",
                            "semantic links of Imagenet labels, it is possible to use ...",
                            "transfer learning to efficiently retrain networks (here VGG16) ...",
                            "to categorise a label of interest (here 'animal')."]),
                   dict(figure='figures_video/robustness_imagenet.png', duration=8, subtitle=[
                            "Results demonstrated a very accurate response over 99%.",
                            "However, as soon as we added a perturbation,",
                            "such as a rotation of the input images...",
                            "the performance significatively dropped."]),
                   dict(figure='figures_video/full_rot.png', duration=10, subtitle=[
                            "The result for this network depends on the rotation,",
                            "angle (VGG LUT). Using data augmentation,",
                            "we could retrain the network to achieve...",
                            "a robust performance for all angles."])
                    ]),

           dict(title="Animal features", color='red',
            content=[dict(figure='figures_video/pruning_accuracy.png', duration=10, subtitle=[
                            "Once this ""ecological"" categorization task defined,...",
                           "a second part of the study was to study the features...",
                           "that explain this robustness. By reducing the depth,...",
                           "of the CNN, and thus the complexity of the features used,...",
                           "we show their role in categorizing an animal...",]),
                    dict(figure='figures_video/full_rot_pruned.png', duration=6, subtitle=[
                            "In particular this gives an idea of the features needed ...",
                           "to perform this task, while retaining the robustness...",
                           " which is also observed in human vision."])]), 
           ]


# http://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html?highlight=compositevideoclip#textclip
txt_opts = dict(fontsize=65, bg_color='white', align='center', **opt_st)
sub_opts = dict(fontsize=28, align='South', color='white', **opt_st)

for chapter in chapters:
    duration = 1.5
    txt = TextClip(chapter['title'], color=chapter['color'], **txt_opts).set_start(t).set_duration(duration)
    t += duration
    clip.append(txt)

    for content in chapter['content']:
        # set the figure
        figname = content['figure']
        duration = content['duration']
        if figname[-4:]=='.mp4' :
            img = VideoFileClip(figname, audio=False)
            print(figname, '--> duration:', img.duration, ', fps:', img.fps)
            # duration = img.duration
        else:
            img = ImageClip(figname)
        
        H_clip, W_clip, three = img.get_frame(0).shape
        if H_clip/W_clip > H_fig/W_fig: # portrait-like
            img = img.resize(height=H_fig)
        else: # landscape-like
            img = img.resize(width=W_fig)

        img = img.set_duration(duration)
        img = img.set_start(t).set_pos('center')#.resize(height=H_fig, width=W_fig)

        t += duration
        clip.append(img)

        # write the subtitles
        t_sub = t - duration
        sub_duration = duration / len(content['subtitle'])
        for subtitle in content['subtitle']:
            sub = TextClip(subtitle, **sub_opts).set_start(t_sub).set_duration(sub_duration)
            t_sub += sub_duration
            clip.append(sub)

#################################################################################
#################################################################################


#################################################################################
# OUTRO
#################################################################################
texts = ["""
Overall, in this paper, we have shown that 
we can re-train networks using transfer learning to
apply them to an ecological image categorization 
task and obtain insights on visuo-cognitive processes.
"""
]

txt_opts = dict(fontsize=30, align='center', **opt_t)
duration = 7
for text in texts:
    txt = TextClip(text, color='white', **txt_opts).set_start(t).set_duration(duration)
    t += duration
    clip.append(txt)
    
# FIN
texts = ["""
For more info, and the full, open-sourced code... visit

""", """
For more info, and the full, open-sourced code... visit
https://laurentperrinet.github.io/publication/jeremie-23-ultra-fast-cat/
""",
]

txt_opts = dict(align='center', **opt_t)
duration = 2.5
for text in texts:
    txt = TextClip(text, color='orange', fontsize=26, **txt_opts).set_start(t).set_duration(duration)
    t += duration
    clip.append(txt)    

# QRCODE
# qrencode -o figures_video/qr_code.png -d 200 https://laurentperrinet.github.io/publication/jeremie-23-ultra-fast-cat/
img = ImageClip('figures_video/qr_code.png').set_duration(duration)
img = img.resize(width=(W_fig*2)//3).set_start(t).set_pos('center')
clip.append(img)

#################################################################################
# COMPOSITING
#################################################################################

video = CompositeVideoClip(clip)
video.write_videofile(videoname + '.mp4', fps=fps)

if not(gifname is None):
    video.write_gif(gifname, fps=fps)
    from pygifsicle import optimize
    optimize(gifname)