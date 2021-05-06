from src.output_visualization.InteractiveWindowSlider import InteractiveWindow

image_path_prefix = r"C:\Users\cernil\OneDrive - Y Soft Corporation " \
                    r"a.s\betapresentation_visualisation\outputs\detection"
image_path_postfix = r"detection.png"
list_of_images = []
n_stages = 15

for stage in range(n_stages):
    image_path = image_path_prefix + "\\" + str(stage + 1) + image_path_postfix
    list_of_images.append(image_path)

InteractiveWindow(list_of_images, None, None, paths_given=True)
