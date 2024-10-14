import gradio as gr
import os

def KAI(input_image, options, MLConfig, facialImgDir):
    print(input_image)
    # create output_img variable by adding KPT to the end of the input_img which could be jpg, jpeg, or png
    if input_image.endswith(".jpg"):
        output_img = input_image.replace(".jpg", "_KAI.jpg")
    elif input_image.endswith(".jpeg"):
        output_img = input_image.replace(".jpeg", "_KAI.jpeg")
    elif input_image.endswith(".png"):
        output_img = input_image.replace(".png", "_KAI.png")
    elif input_image.endswith(".tif"):
        output_img = input_image.replace(".tif", "_KAI.tif")
    print(output_img)
    
    checked_options = []
    # Iterate through the options and add them to the list
    for option in options:
        checked_options.append(option)
        if option == "-facialImgDir":
            checked_options.append(facialImgDir)
    
    # Join the options with space
    checked_options_str = " ".join(checked_options)
    print(checked_options_str)
    
    # print MLConfig file path
    print(MLConfig)

    os.chdir('build')
    # Define the command and arguments
    command = "./KAI-impl " + input_image + " " + MLConfig + \
            " " + output_img + " " + checked_options_str
    os.system(command)
    # go back to the previous directory
    os.chdir('..')
    
    return output_img

demo = gr.Interface(
    fn = KAI,
    inputs = [gr.Image(type="filepath"), 
        gr.CheckboxGroup(["-facialImgDir", "-MLConfig"],
        label = "KAI Options"),
        gr.Textbox(label = "(required) Path to ML Task Configuration File"),
        gr.Textbox(label = "(optional) Path to Facial Imaging library")
    ],
    outputs = gr.Image(type="filepath")
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)