# create the data by the model
all the creation data need to be by model on my repo, that we can fine-tune it.
to do it, you need to download the model to my repo, and create the data by this model!
this is the model we will create the data with: avreymi/gemma-4-E2B-it-reasoning-pruning
by this model we will create the data, then we will train the data on the reasoning traces that we pruning.

make sure this point is clear in the AGENT.md and in every documentation.
then use the model to create the data.


# create interface to paly with promts of pruneing the data
I want this data so I can play with the prompt of the D model and see what is the results.
1. downlad the avreymi/reasoning-traces-gemma4-100 data set, so we can play with this.
2. I want to be the option to create a small chanck of data, localy from the reasoning-traces (only by call the D model, we don't need to call the G model), clearly the D model need to run on claude, don't run it localy. and see the results-how the output will see dicrctly. so we can play with the prompt that create the data and find some good prompt.


you right. it's not good, to create data from depth-0, it's not good, becouse we need to see how the full resolte of creating data look like. 
but the problem is that we cann't run it localy, we need form one hend to handel the G model on fly - so we can change the prompt and see the resulte queicly. but we cannn't do it localy.
what you think? how we can do that?

ok, what I think
we can run it on google-colab, I have subscription and we have enough memroy and gpu to run the G model.
what I will do is: call git to download the project, then import the functoins form the src code. in this way we can play with spesipc part of the code, in this exmple - with the creation data.
what you need:
first maybe you need to change the code. the functions need to be modulry so we can run them out of the scr code file. on the jupyterNotebook - this is very important. I dno't want to create differet functions to paly with data, and differet funtions to create the real data (if you need, you can add one funtion that make interface, but the real functoin under the hood most be the true functoin we create by he the data)
after make sure the source code of create data is build good, and you can export the functions that creat the data out to the jupyterNotebook. write the jupyterNotebook (I add for you the basis - how to pull the git and how to import the a functoin) so we can run the prosees of create data on jupyternotebook see the resulte imdeatly
remember to update all relevent docs 


# Make the results detemnistic
The idea of this project is to train a model to pruning-reasening-traces that it create by self.
to do this, I want to determine a seed, so when we make train the model, is on the same seed. is this possible?



# Pruning Data as Learning

At this stage, the pruning data does not seem good enough, especially for Gemma. The current setup often does not find much useful reasoning to improve or remove.

I want to reframe pruning data as a way to teach the model learned tool use. The idea is to give the agent a complicated task that requires using a tool or skill step by step. We assume the agent initially needs an explicit reasoning trace to use the tool correctly. Then we remove parts of that reasoning trace one sentence at a time, so the model learns to perform the tool-use behavior more automatically.

For this, we need to find or create a dataset focused on tool use, skills, or agent workflows, and then test the pruning method on that kind of data.

My current goal is to remove exactly one sentence at a time. The decision model should find the first sentence in the reasoning trace that is least necessary. I do not want the model to answer that all sentences are necessary. For every reasoning trace, I expect the decision model to identify at least one sentence that can be removed while still preserving the ability to continue correctly.

# Use Qualitative Tests Instead of Only Quantitative Tests

At this stage, the most important test for this project is not a normal pass/fail unit test. We need a qualitative inspection test: a script or notebook-style file that runs the full pruning-data process and prints every step clearly.

The goal is to see how the process actually behaves, not only whether the code runs. We should be able to inspect:

- the original question;
- the context before generation;
- G's generated reasoning trace;
- the split reasoning units;
- D's pruning decision;
- the removed sentence;
- the selected target sentence;
- the final `input_x -> target_y` training row;
- the next context used for the following depth.

This will let us decide whether the data creation process is good, whether the prompt is working, and whether each step matches what we expect.

The problem is that we cannot use the real G model from the repo for this kind of frequent qualitative testing, because it is too expensive to run. Instead, we need to find a cheaper model that is similar enough to G for inspection purposes. For example, there may be a way to run `gemma-4-E2B-it` or a similar Gemma model through an API from Google or Hugging Face.

The task is to research the best practical option for running a model similar to G, without using the real fine-tuned G model. Then build a qualitative test file that runs the same process we used in the notebook and prints each stage of the pipeline.

This test is important because it helps us evaluate the quality of the pruning process itself. We need to see whether each step is producing what we expect, not just whether the code technically works.


# structure results from the D model. - run by codex
use fix the D model to use structure results, it means to use the option that return the results as json.
search in the internet what is the best way to do this.


# simplify the code
make the code simple, the code need to be modular the functoins need to be simple, clear, and intuative! think about every function how to make it intuitive as possible.

# Change the unit split to be more aggressive.
If there is a comma, linking words.
I want a much, much wider division into units, so that the driver has more to download when he needs to.

# remove part of sentences.
In this approach we don't remove whole sentences. Instead, we rewrite the sentence by removing part and add change only connection words. this change requires to change the prompt, and remove the unit split. now we only have sentences and we want to rewrite it shortly by removing part of the sentence. and make only, only small changes. the rational behind it, is that small changes is semanticly closer to the original sentence, and the fine tuning will be work faster and beeter.


# update the observation system - 
this is a research and development project. the code change repidly, and the most important thing is to find the best pipline from createing the data, to the training and evaluation. to do this we most have a good observation system. and the prosses of update the code (if isn't only change the code that not effects on the pipeline) most have in addition to the tests that the code works. you have to use the observation system to see how this changes effect the pipline. so we have to look every step and think about this and make qualitative inspection for every step. for this we have the notebook and the scripts qualitative_inspection.
I need you update every docs about this point, emphasize the role of the project as research and development, and the importance of the observation system/qualitative_inspection, and the importance of use it.
pass trouge AGENTS.md and skills and update this point. I want to change the way we think about this project.
also - we use two names: observation system and qualitative inspection, I want to unify it to one name. Make sure the terminology in the documentation and code is consistent and clear about what you're talking about. You can change names or whatever you want.


# Allow agents using pipeline inspection by colab
now, when we need to use the pipeline inspection, we use the notebook /home/avreymi/code/reasoning-pruning-codex/notebooks/data_creation_playground.ipynb, for example when we need to play with the unit_split_strategy and D prompt. I want to allow agents to do this. but the problem is that we can run the notebook only on colab server (becouse we need at least T4 high ram to run it).
I added colab-mcp and colab-cli to work with colab. the colab-cli is new, you have to read lot about it.
What I need is:
read the colab-cli documentation, try to run the mcp, and add skill about using colab. focus on interactive interface that allow us start a server, inspect the pipeline change it live and continue. without the need to stop the server and start it again. this is very important, because we want to play with the prompt and see the results immediately, without the need to stop and start the server again. also focus on runing the notebook.
the results sholde be written to the output/pipeline_inspection.
play with this, make sure you understand it good. and create the skill.
remeber focus on:
1 - how to use by agents.
2 - how make pipeline inspection by this.


1. the project isnot only for gemma4, we want to investigate variety of models.
2. The agent need to run things close to the way it runs in the notebook, maybe run cells from the notebook? (only if it not make the code more complicated)
3. use the mpc is good idea. I open the colab notebook, play with it. here the link:https://colab.research.google.com/github/avrymi-asraf/reasoning-pruning/blob/master/notebooks/data_creation_playground.ipynb. if you need more things let my know.