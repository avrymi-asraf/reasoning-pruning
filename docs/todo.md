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


