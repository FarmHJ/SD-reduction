import modelling

# Define AP model, drug and tuning method
drug_list = modelling.data.drug_names
dataset = modelling.DataLibrary()

###################
# Experimental data
###################

for drug in drug_list:

    print(drug)
    dataset.set_drug(drug)
    dataset.plot_signal()
    dataset.plot_mean_signal()
