from __future__ import print_function
import numpy as np

def eval_score_table(score_table, ranks, row_IDs, col_IDs):  
    row_IDs = np.array(row_IDs)
    col_IDs = np.array(col_IDs)

    num_col = col_IDs.shape[0]
    num_row = row_IDs.shape[0]

    # verify score_table size against row_IDs and col_IDs
    #&&&
      
    # verify ranks agains score_table size 
    #ranks = [rank for rank in ranks if rank <= num_samples] # filter bad ranks
    accuracy = [0.] * len(ranks)    
      
    sorted_idx = np.argsort(score_table, axis=1)
  
    for r, rank in enumerate(ranks):
        num_matches = 0
        for i in range(num_row):
            sorted_row_ids = col_IDs[sorted_idx[i]]
            if row_IDs[i] in sorted_row_ids[-rank:]:
                num_matches += 1
        accuracy[r] = (100 * num_matches)/float(num_row)
        
    print('Rank based accuracy:')
    for i in range(len(ranks)):
        print('Top {0:3d} : {1:2.2f}%'.format(ranks[i], accuracy[i]))
        
    #for i in range(num_row):
    #    print(row_IDs[i], 'Scores ')
    #    print(col_IDs[sorted_idx[i]])
    #    print(score_table[i][sorted_idx[i]])