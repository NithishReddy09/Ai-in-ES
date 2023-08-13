# Import the necessary libraries
from transformers import pipeline
import gradio as gr
from gradio.mix import Parallel, Series

# Introduction for the summarizer
intro = """BART Summarizer"""

# Sample paragraphs
s1 = """Deep within the heart of the dense forest, a group of explorers embarked on an expedition that would unveil the secrets of a hidden civilization. As they delved deeper into the wilderness, they discovered ancient ruins adorned with intricate carvings, telling stories of a people who had harmoniously coexisted with nature. The expedition became a journey of deciphering the past, connecting the present with a forgotten legacy of sustainable living and ecological wisdom."""

s2 = """In the remote valleys of a mountainous region, a community of artists and inventors gathered to forge a new way of blending tradition with innovation. Through intricate artwork and craftsmanship, they wove modern concepts into traditional tapestries, pottery, and sculptures, creating a living chronicle of their time. This convergence of old and new served as a reminder that progress can be a harmonious dance, where the steps of the past guide the leaps of the future."""

s3 = """On the vast plains of a changing world, a group of nomads traversed the landscapes with a profound connection to the rhythms of nature. They practiced sustainable farming techniques handed down through generations, using the land's natural cycles to cultivate crops. Their migratory lifestyle left minimal impact, a testament to their ethos of leaving only footprints. In an era of rapid change, they stood as guardians of ancient knowledge, proving that harmony with the Earth is a timeless journey."""

# Organizing sample paragraphs into a list
sample = [[s1],[s2],[s3]]

# Loading the BART summarization model using the Hugging Face Transformers library
io = gr.Interface.load("huggingface/facebook/bart-large-cnn")

# Configuring the Gradio interface
iface = Parallel(io,
                 theme='huggingface', 
                 title= 'Summarizer', 
                 description = intro,
                 examples=sample,
                 inputs = gr.inputs.Textbox(lines = 10, label="Text"))

# Launching the Gradio interface
iface.launch(inline = False, enable_queue = True, show_api = False)

