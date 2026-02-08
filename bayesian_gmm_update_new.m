function [new_gmm, task_bayes, mu_new] = bayesian_gmm_update_new(task_bayes, valid_M, prior_available, weights)
    %% 参数设置
    if nargin < 4 || isempty(weights)
        weights = ones(size(valid_M, 1), 1); 
    end
    weights = weights(:);

    prior_uncertainty = task_bayes.prior_uncertainty;
    K = task_bayes.K;
    
    beta = ones(1,K)*0.05;    
    alpha = 1.0; epsilon = 1e-4; min_pi = 1e-5;
    
    [N, dim] = size(valid_M); % 获取样本数 N 和维度 dim

    %% 初始化逻辑 (修正版)
    if isempty(task_bayes.gmm)
        % 1. 初始化协方差
        % [修正] 专门处理 N=1 的情况，防止 cov 返回标量或 NaN
        if N <= 1
            Sigma_init = eye(dim) * epsilon;
        else
            Sigma_init = cov(valid_M);
            % 再次检查，防止协方差数值异常
            if any(isnan(Sigma_init(:))) || any(isinf(Sigma_init(:)))
                Sigma_init = eye(dim) * epsilon;
            end
        end
        
        % 构建 K 个协方差矩阵 (d x d x K)
        Sigma_array = arrayfun(@(k) make_spd(Sigma_init, epsilon), 1:K, 'UniformOutput', false);
        Sigma_all = cat(3, Sigma_array{:});
        
        % 2. 初始化均值 (C)
        try
            if N >= K
                % 样本充足时使用 K-Means
                [~, C] = kmeans(valid_M, K, 'Replicates', 3, 'EmptyAction', 'singleton');
            else
                error('Not enough samples for K-means');
            end
        catch
            % [修正] 回退策略：随机采样 + 均值填充
            % 如果样本不够，先取所有样本
            rand_idx = randperm(N, min(N, K));
            C = valid_M(rand_idx, :);
            
            % 如果还不够 K 个，用整体均值填充剩余的
            if size(C, 1) < K
                % [关键修正] 必须加 ,1 强制按列求均值，防止 N=1 时 mean 返回标量导致 vertcat 失败
                mean_vec = mean(valid_M, 1); 
                C = [C; repmat(mean_vec, K - size(C, 1), 1)];
            end
        end
        
        % 3. 注入专家先验
        if prior_available
            prior_mu = task_bayes.prior_mu;
            for i = 1:min(size(prior_mu,1), K)
                C(i,:) = prior_mu(i,:);
                Sigma_all(:,:,i) = eye(dim) * (prior_uncertainty^2);
            end
        end
        
        % 构建 GMM 对象
        try
            task_bayes.gmm = gmdistribution(C, Sigma_all, ones(1,K)/K);
        catch ME
            % 如果构建失败，输出调试信息并尝试强制修复 Sigma
            warning('GMM init failed: %s. Re-initializing with identity covariance.', ME.message);
            Sigma_safe = repmat(eye(dim) * epsilon, [1, 1, K]);
            task_bayes.gmm = gmdistribution(C, Sigma_safe, ones(1,K)/K);
        end
    end
    
    %% 变分EM迭代 (后续代码保持不变，直接复制即可)
    % ... (请保留之前提供的 EM 迭代部分) ...
    
    current_gmm = task_bayes.gmm;
    % [这里直接接续之前的 EM 代码逻辑，没有变动]
    
    % --- 为了完整性，这里重复 EM 核心部分 ---
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