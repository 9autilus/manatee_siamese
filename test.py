from __future__ import print_function

def test_single_source(model, X, ID):
    print('Computing rank-based accuracy... ')
    num_samples = X.shape[0]
    num_pairs = (num_samples * (num_samples + 1)) / 2

    ranks = sorted([1, 5, 10, 20])
    ranks = [rank for rank in ranks if rank <= num_samples] # filter bad ranks
    
    size = 1
    for i in [num_pairs, 2] + [i for i in X[0].shape]:
        size *= i
    
    # To tackle memory constraints, process pairs in small sized batchs
    batch_size = 32;    # Num pairs in a batch
    scores = np.empty(num_pairs).astype('float32')         # List to store scores of ALL pairs
    
    pairs = np.empty([batch_size, 2] + [i for i in X[0].shape]);
    pair_count = 0; counter = 0;
    for i in range(num_pairs):
        for j in range(i, num_samples):
            pairs[pair_count] = np.array([[X[i], X[j]]])
            pair_count += 1
            
            if (pair_count == batch_size):
                print('\r Predicting {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                batch_scores = model.predict([pairs[:, 0], pairs[:, 1]], batch_size=batch_size)
                scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                pair_count = 0
                counter += batch_size
                
    # Last remaining pairs which couldn't get processed in main loop
    if pair_count > 0:
        batch_scores = model.predict([pairs[:pair_count, 0], pairs[:pair_count, 1]])
        scores[counter:(counter+pair_count)] = batch_scores[:, 0]
    
    print('Analyzing scores... ', end='')
    score_table = np.zeros([num_samples, num_samples]).astype('float32')
    idx = 0
    for i in range(num_samples):
        for j in range(i, num_samples):
            score_table[i, j] = scores[idx]
            idx = idx + 1
    score_table += np.transpose(score_table)
    for i in range(num_samples):
        score_table[i][i] /= 2.
        
    eval_score_table(score_table, ranks, ID, ID)
        
def perform_testing(model, X1, ID1, X2=None, ID2=None):
    test_single_source = (X2 is None) or (X1 is None)

    print('Computing rank-based accuracy... ')
    num_rows = X1.shape[0]
    
    if test_single_source:
        X2 = X1; ID2 = ID1
        num_cols = num_rows
        num_pairs = (num_rows * (num_rows + 1)) / 2
    else:
        num_cols = X2.shape[0]
        num_pairs = num_rows * num_cols
    
    ranks = sorted([1, 5, 10, 20])
    ranks = [rank for rank in ranks if rank <= num_cols] # filter bad ranks
    
    # To tackle memory constraints, process pairs in small sized batchs
    batch_size = 32;    # Num pairs in a batch
    scores = np.empty(num_pairs).astype('float32') # List to store scores of ALL pairs     
    score_table = np.zeros([num_rows, num_cols]).astype('float32')
    
    if test_single_source:
        pairs = np.empty([batch_size, 2] + [i for i in X1[0].shape]);
        pair_count = 0; counter = 0;
        for i in range(num_rows):
            for j in range(i, num_cols):
                pairs[pair_count] = np.array([[X1[i], X1[j]]])
                pair_count += 1
                
                if (pair_count == batch_size):
                    print('\r Predicting pair {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                    batch_scores = model.predict([pairs[:, 0], pairs[:, 1]], batch_size=batch_size)
                    scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                    pair_count = 0
                    counter += batch_size
                    
        # Last remaining pairs which couldn't get processed in main loop
        if pair_count > 0:
            print('\r Predicting pair {0:d}/{1:d}'.format(num_pairs, num_pairs), end=''); sys.stdout.flush()
            batch_scores = model.predict([pairs[:pair_count, 0], pairs[:pair_count, 1]])
            scores[counter:(counter+pair_count)] = batch_scores[:, 0]
            
        print('Analyzing scores... ', end='')
        idx = 0
        for i in range(num_rows):
            for j in range(i, num_cols):
                score_table[i, j] = scores[idx]
                idx = idx + 1
        score_table += np.transpose(score_table)
        for i in range(num_rows):
            score_table[i][i] /= 2.
    else:   
        pairs = np.empty([batch_size, 2] + [i for i in X1[0].shape]);
        pair_count = 0; counter = 0;
        for i in range(num_rows):
            for j in range(num_cols):
                pairs[pair_count] = np.array([[X1[i], X2[j]]])
                pair_count += 1
                
                if (pair_count == batch_size):
                    print('\r Predicting pair {0:d}/{1:d}'.format(counter, num_pairs), end=''); sys.stdout.flush()
                    batch_scores = model.predict([pairs[:, 0], pairs[:, 1]], batch_size=batch_size)
                    scores[counter:(counter+pair_count)] = batch_scores[:, 0]
                    pair_count = 0
                    counter += batch_size
                    
        # Last remaining pairs which couldn't get processed in main loop
        if pair_count > 0:
            print('\r Predicting pair {0:d}/{1:d}'.format(num_pairs, num_pairs), end=''); sys.stdout.flush()
            batch_scores = model.predict([pairs[:pair_count, 0], pairs[:pair_count, 1]])
            scores[counter:(counter+pair_count)] = batch_scores[:, 0]
            
        print('Analyzing scores... ', end='')
        idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                score_table[i, j] = scores[idx]
                idx = idx + 1         
    
    # Parse score table and generate accuracy metrics
    eval_score_table(score_table, ranks, ID1, ID2)        
        

        

    
def debug_sketches(X1, X2, ht, wd) :
    for i in range(X1.shape[0]):
        print(np.min(X1[i]), np.mean(X1[i]), np.max(X1[i]))
    
    for i in range(X1.shape[0]):
        sketch = X1[i].reshape(ht, wd)
        sketch = sketch - np.min(sketch)
        sketch = sketch * 255/np.max(sketch)
        sketch = sketch.astype('uint8')
        file_name = 'test2' + str(i) + '_' + str(wd) + 'x' + str(ht)+ '.png'
        cv2.imwrite(file_name, sketch)
    
def test_net(args):
    input_dim = (1, ht, wd)
    # network definition
    model = create_network(input_dim)

    # Reuse pre-trained weights
    print('Reading weights from disk: ', args.weights)
    #model.load_weights(args.weights)
    
    if args.test_mode == 0:
        X, ID = load_sketches(test_dir)
        perform_testing(model, X, ID)
    elif args.test_mode == 1:
        X1, ID1 = load_sketches(test_dir)
        X2, ID2 = load_sketches(train_dir)
        X2 = np.concatenate((X1, X2), axis=0);
        ID2 = np.concatenate((ID1, ID2), axis=0);
        perform_testing(model, X1, ID1, X2, ID2)
    elif args.test_mode == 2:
        X1, ID1 = load_sketches(test2_dir)
        X2_1, ID2_1 = load_sketches(train_dir)
        X2_2, ID2_2 = load_sketches(test_dir)
        X2 = np.concatenate((X2_1, X2_2), axis=0);
        ID2 = np.concatenate((ID2_1, ID2_2), axis=0);
        print(X1.shape, X2_1.shape, X2_2.shape, X2.shape)

    debug_sketches(X1, X2, ht, wd)    
    #perform_testing(model, X1, ID1, X2, ID2)   