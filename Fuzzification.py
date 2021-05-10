import numpy as np

class Fuzzification:
    def __init__(self, row, mean, sd):
        self.row = row #each row with multiple features
        self.mean = mean #means from dynamic clustering for each features
        self.sd = sd #sds from dynamic clustering for each features
        
    def normal_density(self, x, mean, sd):
        if sd == 0:
            if x == mean:
                return 1
            else:
                return 0
        return (1/(sd*(2*np.pi))**(0.5))* np.exp(-0.5*((x-mean)/sd)**(2))
    
    def fuzzify(self):
        if len(self.row) != len(self.mean):
            raise Exception('Need means for each feature')
        if len(self.row) != len(self.sd):
            raise Exception('Need sd for each feature')
        fuzzy_output = []    
        for i in range(len(self.row)):           
            clusters = len(self.mean[i])
            output = np.zeros((clusters))
            densities = []
            for j in range(clusters):
                density = self.normal_density(self.row[i], self.mean[i][j], self.sd[i][j])
                densities.append(density)
        
            index = densities.index(max(densities))
            output[index] = 1
            output = output.tolist()
            output = [int(x) for x in output]
            fuzzy_output.append(output)
            #fuzzy_output.append()
        return fuzzy_output