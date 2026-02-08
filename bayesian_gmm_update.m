function [new_gmm, task_bayes, mu_new] = bayesian_gmm_update(task_bayes, valid_M, prior_available, weights)
    if nargin < 4 || isempty(weights)
        weights = ones(size(valid_M, 1), 1); 
    end
    weights = weights(:);

    prior_uncertainty = task_bayes.prior_uncertainty;
    K = task_bayes.K;
    
    beta = ones(1,K)*0.05;    
    alpha = 1.0; epsilon = 1e-4; min_pi = 1e-5;
    
    [N, dim] = size(valid_M); 

    if isempty(task_bayes.gmm)
        if N <= 1
            Sigma_init = eye(dim) * epsilon;
        else
            Sigma_init = cov(valid_M);
            if any(isnan(Sigma_init(:))) || any(isinf(Sigma_init(:)))
                Sigma_init = eye(dim) * epsilon;
            end
        end
        
        Sigma_array = arrayfun(@(k) make_spd(Sigma_init, epsilon), 1:K, 'UniformOutput', false);
        Sigma_all = cat(3, Sigma_array{:});
        try
            if N >= K
                [~, C] = kmeans(valid_M, K, 'Replicates', 3, 'EmptyAction', 'singleton');
            else
                error('Not enough samples for K-means');
            end
        catch
            rand_idx = randperm(N, min(N, K));
            C = valid_M(rand_idx, :);
            
            if size(C, 1) < K
                mean_vec = mean(valid_M, 1); 
                C = [C; repmat(mean_vec, K - size(C, 1), 1)];
            end
        end
        
        if prior_available
            prior_mu = task_bayes.prior_mu;
            for i = 1:min(size(prior_mu,1), K)
                C(i,:) = prior_mu(i,:);
                Sigma_all(:,:,i) = eye(dim) * (prior_uncertainty^2);
            end
        end
        
        try
            task_bayes.gmm = gmdistribution(C, Sigma_all, ones(1,K)/K);
        catch ME
            warning('GMM init failed: %s. Re-initializing with identity covariance.', ME.message);
            Sigma_safe = repmat(eye(dim) * epsilon, [1, 1, K]);
            task_bayes.gmm = gmdistribution(C, Sigma_safe, ones(1,K)/K);
        end
    end
    
    current_gmm = task_bayes.gmm;
    mu_prior = current_gmm.mu;
    try
        gamma = posterior(current_gmm, valid_M);
    catch
        dists = pdist2(valid_M, current_gmm.mu);
        [~, min_idx] = min(dists, [], 2);
        gamma = zeros(N, K);
        for i = 1:N
            gamma(i, min_idx(i)) = 1;
        end
    end
    
    effective_gamma = gamma .* weights;
    Nk_effective = sum(effective_gamma, 1);
    Total_Weight = sum(weights); 
    
    pi_new = (Nk_effective + alpha) / (Total_Weight + K * alpha);
    pi_new = max(pi_new, min_pi);
    pi_new = pi_new / sum(pi_new);
    
    mu_new = zeros(K, dim);
    Sigma_new = zeros(dim, dim, K);
    
    for k = 1:K
        weighted_sum_data = effective_gamma(:, k)' * valid_M;
        mu_new(k, :) = (weighted_sum_data + beta(k) * mu_prior(k, :)) / (Nk_effective(k) + beta(k));
        
        diffs = valid_M - mu_new(k, :);
        sqrt_weights = sqrt(effective_gamma(:, k));
        weighted_diffs = diffs .* sqrt_weights;
        scatter_data = weighted_diffs' * weighted_diffs;
        
        diff_prior = mu_prior(k, :) - mu_new(k, :);
        scatter_prior = beta(k) * (diff_prior' * diff_prior);
        
        denominator = Nk_effective(k) + beta(k) + dim + 1;
        Sigma_k = (scatter_data + scatter_prior) / denominator;
        
        Sigma_k = make_spd(Sigma_k, epsilon);
        strict_positive_threshold = max(epsilon, 1e-8); 
        if min(eig(Sigma_k)) < strict_positive_threshold
            Sigma_k = Sigma_k + strict_positive_threshold * eye(size(Sigma_k));
            Sigma_k = (Sigma_k + Sigma_k') / 2;
        end
        Sigma_new(:, :, k) = Sigma_k;
    end
    
    try
        new_gmm = gmdistribution(mu_new, Sigma_new, pi_new);
    catch
        new_gmm = task_bayes.gmm; % 回退
        mu_new = task_bayes.gmm.mu;
    end
end

function A_spd = make_spd(A, epsilon)
    A = (A + A') / 2;
    [V, D] = eig(A);
    d = diag(D);
    d(d < epsilon) = epsilon;
    A_spd = V * diag(d) * V';
    A_spd = (A_spd + A_spd') / 2;
end
