 function centroids = initCentroids(K, X)
 
    centroids = zeros(K,size(X,2)); 
    
    % Randomly reorder the indices of examples
    randidx = randperm(size(X,1));
     
    % Take the first K examples as centroids
    centroids = X(randidx(1:K), :);
    
  end