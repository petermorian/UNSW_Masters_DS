# Stage 1: Reading and writing data to and from files
def read_from_file(filename):
    given_file = open(filename,"r").readlines() #reads each line in file
    list_of_floats = [float(i) for i in given_file if i != "-\n"] #floats each element in list, excluding "-"
    return list_of_floats 

def write_to_file(filename, binned_data):
    new_file = open(filename,"w") #load file and create new lines for each dictionary row
    new_file.write("\n".join('{}: {}'.format(key, value) for key, value in binned_data.items()))



# Stage 2: Place the data into bins
def bin_data(data, binsize):
    
     #CREATE BIN BOUNDS
    bin_start = min(data) #lowest value of data
    bin_end = max(data) #highest value of data
    bins = [bin_start] #first bin limit is the lowest valye
    next_bin = bin_start
    while next_bin <= bin_end: #making sure each upper bound is not the max
        next_bin = round(next_bin + binsize,2) #each upper limit is created by adding the float 'binsize'
        bins.append(next_bin)
    
    # CREATE BIN RANGES FOR DICTIONARY KEY
    lower_bounds = bins[:-1] #Left side of bins
    upper_bounds = bins[1:] #Right side of bins
    bin_ranges = [str(a)+" - "+str(b) for a,b in zip(lower_bounds,upper_bounds)] #create keys
    
    # CREATE COUNTS FOR DICTIONARY VALUES
    from collections import Counter
    counts = Counter() #create empty dictionary of counts
    for j in range(0,(len(bin_ranges))): #for each bin range
        for i in range(0,(len(data))): #for each data point
            if data[i] >= lower_bounds[j] and data[i] < upper_bounds[j]:
                counts[j] += 1 #if data is between the bounds, increase count
            else:
                counts[j] += 0 #if data is not between bounds, add 0 to count

    #CREATE DICTIONARY WITH BIN KEYS & COUNT VALUES
    bin_dict = dict((bin_ranges[key], value) for (key, value) in counts.items()) #replace key from Counter function with bins
    return bin_dict


# Stage 3: Display the frequency analysis
def display_freq_analysis(binned_data):
    for i in binned_data: #print each key and value separately from dictionary
        print(i,": ",binned_data[i], sep="") 

    

# Stages 4 and 5: Interact with the user with menus and error handling.
# The following line suppresses the code when the file is imported 
import sys

if __name__ == '__main__':
    
    selected_file=input('Enter data file: ') 
    while selected_file not in ("data01.txt", "data02.txt", "data03.txt"):
        print("File not available") #only work for data01/02/03 input
        selected_file=input('Enter data file: ') 
    loaded_file = read_from_file(selected_file) #run Stage 1 write function
    
    choice = True
    while choice:
        print("1. Set bin size\n2. Display frequency analysis\n3. Exit")
        choice=input('Please enter your choice: ')
        
        try:
            if int(choice) == 1:
                bin_size=float(input("Enter bin size: ")) #bin size must be numeric
                try:
                    float(bin_size)
                except ValueError:
                    print("Bin size must be numeric. Try again.")
                    
            elif int(choice) == 2: #create/display bins & save file
                try:
                    binned_selected_data = bin_data(loaded_file, bin_size) #runs Stage 2 function
                    display_freq_analysis(binned_selected_data) #run Stage 3 function
                    save_file= input("Save analysis to file? (Y/N) ")
                    while save_file not in ("Y","N"): #reask question if invalid response
                        print("Not a valid option")
                        save_file= input("Save analysis to file? (Y/N) ")
                    if save_file == "Y":
                        file_save_name = input("Enter file name: ")
                        write_to_file(file_save_name, binned_selected_data) #run Stage 1 save function
                except NameError:
                    print("Please set a bin size first") #return to menu if choice 2 was selected before choice 1 
                
            elif int(choice) == 3: #exit program if choice 3
                sys.exit()
            
            else: # return to menu if invalid choice, but input is a integer not 1, 2 or 3
                print("Not a valid option")

        except ValueError: # return to menu if invalid choice, but input is not an integer
            print("Not a valid option")
            