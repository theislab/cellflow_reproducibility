# I scheduled all CellOT runs with 
# 1_run_all_models.sh drugs.txt

# Then I evalauted them using the lines in 
# 2_evaluation_loop.txt





# I changed the following lines in the CellOT model:

# - line 57 in cellot-main/cellot/train/train.py:

I changed this:
target = next(iterator_test_target)
source = next(iterator_test_source)

to this:
target = next(iterator_train_target)
source = next(iterator_train_source)

# I did this such that it "evaluates" using test data (this is for what model it picks in the end, and by default it picks
# the one with the best evaluation on the test data, which is cheating, so I let it pick the best model on the training data)



# - line 94 in  /cellot-main/cellot/data/cell.py:

I changed this:
# write train/test/valid into split column
data = data[mask].copy()
if "datasplit" in config:
    data.obs["split"] = split_cell_data(data, **config.datasplit)
        
to this:

# write train/test/valid into split column
data = data[mask].copy()
dr = sorted(set(data.obs["drug"]))
dr = [a for a in dr if a!='control']
if len(dr)>1:
    print('Error: More than one drug')
else:
    dr=dr[0]
datasplit_df = pd.read_csv(f'/home/icb/manuel.gander/pert/data/splits/{dr}.csv')
D_datasplit = dict(zip(datasplit_df['Unnamed: 0'], datasplit_df['split']))
data.obs['split'] = data.obs.index.map(D_datasplit)
    
# I did this to load my training/testing split, such that I use the same for all the different models


I added both these files into the CellOT_changed_scripts-folder
