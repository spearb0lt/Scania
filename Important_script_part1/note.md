Data-Processing.ipynb shows step by step approach to clean and make the datasets robust


Preprocess.ipynb does preprocessing straight out train,test,valid all are done


Basic plottings contains some basic plots from thr data




Modelling_part1 -->first time used a TF emc=beddeer for numeric and also saved sequence_embeddings.npy", all_embs)
np.save("sequence_vehicle_ids.npy", all_vids)


LSTM GRU TF CATEGORICAL 1st tiem used in initial_model



Modelling_part2 new recent model with Dynamic+Hardcoded paths also contains diffeent plots regarding model paras and graphs of it

Fresh is made from it


load_saved_model helps to load models and draw inference from it
differential_privacy.ipynb is the first time of combining model+dp



running_tf+dp.py is a single py script doing everything that the above ipynb were doing
running_tf+dp2.0.py is a single py script which is modified from 1.0 it has early stopper and stepLR and its log generation is different