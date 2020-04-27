# load relevant libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.set_option('display.expand_frame_repr', False) #shows full DataFrames in output

# Stage 1: Read and process the data
def read_and_process(filename):
    df=pd.read_csv(filename).dropna() #remove rows with no values in at least 1 column
    measurements=df.iloc[:,:-1].applymap(lambda x: x.rstrip('cmmm')) #removes cm/mm from end of element
    measurements=measurements.astype(float) #measurments as floats
    measurements['sepal_width']=measurements['sepal_width']/10 #divide septal width by 10
    final_df=pd.concat([measurements,df.iloc[:,-1]], axis=1, join='outer', ignore_index=False)
    return final_df

# Stage 5: Conculsion
def conclusion():
    # Return the two (non-species) categories that best identify the species of iris
    features_tuple = ('petal_length', 'petal_width') #base of the plot for all features
    return features_tuple
    

import sys
if __name__ == '__main__':
    
    # Stage 2: User menu
    given_file=input('Enter csv file: ')
    df=read_and_process(given_file) #run stage 1 function
    
    choice = True
    while choice:
        print('1. Create textual analysis\n2. Create graphical analysis\n3. Exit')
        choice=int(input('Please select an option: '))
        try:
            if int(choice) == 1: # Stage 3: Text-based analysis
                species_options=sorted(pd.unique(df.iloc[:,-1].values)) #sorted options of species to filter for
                question_species = 'Select species (all, ' + ', '.join(str(i) for i in species_options).strip() + '): '
                species=input(question_species)
                if species == 'all': # get summary stats for entire dataset 
                    summary_stats=df.describe().T.rename(columns={"mean": "Mean", "50%": "Median", "std": "Std"})
                else: #filter data for selected speicies
                    summary_stats=df.loc[df['species'] == species].describe().T.rename(columns={"mean": "Mean", "50%": "Median", "std": "Std"})
                print(summary_stats[['Mean','25%','Median','75%', 'Std']]) #only display five columns

            elif int(choice) == 2: # Stage 4: Graphics-based analysis
                x_axis_option=input('Choose the x-axis characteristic (all, sepal_length, sepal_width, petal_length, petal_width): ')
                if x_axis_option != 'all': #for a specific x-axis feature
                    y_axis_option=input('Choose the y-axis characteristic (sepal_length, sepal_width, petal_length, petal_width): ')
                    #species_labels=list(set(df['species'])) #collect unique species
                    #for i,u in enumerate(species_labels): #for each species, make a scatter plot with unique colour and label
                     #   x_i=[df[x_axis_option][j] for j in range(len(df[x_axis_option])) if df['species'][j] == u]
                      #  y_i=[df[y_axis_option][j] for j in range(len(df[y_axis_option])) if df['species'][j] == u]
                       # plt.scatter(x_i, y_i, label=str(u))
                    sns.lmplot( x=x_axis_option, y=y_axis_option, data=df, fit_reg=False, hue='species', legend=True)
                    plt.legend(title='Species') #add legend title
                    plt.xlabel(x_axis_option) #add x-axis label
                    plt.ylabel(y_axis_option) #add y-axis label
                else: #plot relationship between all characteristics (different colour per species)
                    sns.pairplot(df, hue='species') #diagonal plots show frequenies per species
                save_file_name= input("Enter save file: ") #save image
                plt.savefig(save_file_name) 

            elif int(choice) == 3:
                sys.exit() #leave menu if choice 3 is entered

            else: # return to menu if invalid choice, but input was an integer not 1, 2 or 3
                print("Not a valid option")

        except ValueError: # return to menu if invalid choice, but input was not an integer
                print("Not a valid option")
