function [task, bayes] = BDA_new(task, bayes, gen)
    % BDA: Bayesian Domain Adaptation with Population-Sum Normalization
    % Corrected: Weights are strictly normalized by population SUM, ensuring w < 1.
    
    %% Common Parameters
    gen_tran = 6;           
    transfer_ratio = 0.2;   
    num_per_sol = 1;        
    prior_available = task(1).prior_available; 
    prior_uncertainty = task(1).prior_uncertainty;
    
    epsilon = 1e-6; % Prevent division by zero

    for n_target = 1:length(task)
        %% Initialization -------------------------------------------
        n_source = setdiff(1:length(task), n_target);
        
        if ~isfield(bayes, ['task' num2str(n_target)])
            bayes.(['task' num2str(n_target)]) = struct();
        end
        task_bayes = bayes.(['task' num2str(n_target)]);

        if gen == gen_tran
            task_bayes.gmm = []; 
            task_bayes.replace_ratio = 0.5; 
            task_bayes.M_history = cell(100,1); 
            task_bayes.trans_fitness = cell(100,1); 
            task_bayes.prior_uncertainty = prior_uncertainty;
            task_bayes.replace_history = []; 
            task_bayes.M_source = cell(100,1); 
            task_bayes.mu{gen} = [];
            
            task_bayes.prior_mu = zeros(bayes.K, size(task(n_target).Lb,2));
            real_shift = task(n_target).x_best - task(n_source).x_best;
            for numn = 1:1
                 [~,~, task_bayes.prior_mu(numn,:)] = uncertainty_generate(real_shift, prior_uncertainty);
            end
            task_bayes.K = bayes.K;
        end

        %% Phase 1: Bayesian Posterior Update (Learning) -----------------------
        if gen > gen_tran
            prev_gen = gen; 
            prev_M = task_bayes.M_history{prev_gen}; 
            prev_fitness = task_bayes.trans_fitness{prev_gen}; 
            prev_source = task_bayes.M_source{prev_gen};

            if ~isempty(prev_fitness) && ~isempty(prev_M)
                
                % [CRITICAL UPDATE]: Likelihood Normalized by Population Sum
                current_pop_fitness = task(n_target).fitness;
                
                % 1. Baseline: Worst surviving individual
                b_t = max(current_pop_fitness); 
                
                % 2. Calculate Population Total Quality (Q_pop)
                % Raw improvement of the population
                pop_improvement = max(0, b_t - current_pop_fitness);
                % Sum of improvements (Denominator)
                Q_pop = sum(pop_improvement);
                
                % 3. Calculate Weights for Transferred Solutions
                % Raw improvement of transferred solutions
                raw_diff = b_t - prev_fitness; 
                scores_raw = max(0, raw_diff);
                
                % Normalize by Q_pop
                % Result: w_i < 1 (Probability-like)
                scores = scores_raw ./ (Q_pop + epsilon);
                
                % Identify effective samples
                valid_idx = find(scores > 1e-10); 
                
                if ~isempty(valid_idx)
                    valid_M = prev_M(valid_idx, :);
                    scaling_factor = length(scores); 
                    valid_scores = scores(valid_idx);
                    % valid_scores = scores(valid_idx) * scaling_factor;
                    
                    % Update GMM with these small probability weights
                    % The sum(valid_scores) will act as N_eff.
                    [updated_gmm, task_bayes, mu_new] = bayesian_gmm_update_new(...
                        task_bayes, valid_M, prior_available, valid_scores); 
                    
                    task_bayes.mu{gen} = mu_new;
                    task_bayes.gmm = updated_gmm;
                    
                    % Adaptive Sampling (Bandit) Update
                    source_labels = prev_source(valid_idx);
                    idx_exist = strcmp(source_labels, 'exist');
                    idx_bayes = strcmp(source_labels, 'bayes');
                    score_exist = valid_scores(idx_exist);
                    score_bayes = valid_scores(idx_bayes);
                    
                    if ~isempty(score_exist) && ~isempty(score_bayes)
                        avg_exist = mean(score_exist); 
                        avg_bayes = mean(score_bayes);
                        score_diff = avg_exist - avg_bayes;
                        adjust_rate = 0.1;
                        task_bayes.replace_ratio = max(0.1, min(0.9, ...
                            task_bayes.replace_ratio + adjust_rate * tanh(score_diff))); 
                    end
                end
                task_bayes.replace_history(end+1) = task_bayes.replace_ratio;
            else
                task_bayes.replace_history(end+1) = task_bayes.replace_ratio;
            end
        else
             if isfield(task_bayes, 'replace_ratio')
                task_bayes.replace_history(end+1) = task_bayes.replace_ratio;
             end
        end

        %% Phase 2: Generate & Inject (Transfer) -------------------
        if gen >= gen_tran
            % (Phase 2 代码逻辑保持不变)
            source_fit = task(n_source).fitnesses{gen};
            source_sol = task(n_source).solutions{gen};
            
            if ~isempty(source_fit)
                [~, rank_idx] = sort(source_fit); 
                elite_num = max(1, round(transfer_ratio * length(source_fit)));
                elite_sol = source_sol(rank_idx(1:elite_num), :);
            else
                elite_sol = [];
            end
            
            M_buffer = []; 
            source_type_buffer = {};
            
            if ~isempty(elite_sol)
                for i = 1:size(elite_sol, 1)
                    for j = 1:num_per_sol
                        if (gen == gen_tran) || (rand() < task_bayes.replace_ratio) || isempty(task_bayes.gmm)
                             M = adaptation_parameter(...
                                task(n_target).solutions, task(n_target).fitnesses, ...
                                task(n_source).solutions, task(n_source).fitnesses, ...
                                elite_sol(i,:), task(n_target).x_best, task(n_source).x_best, ...
                                prior_available, prior_uncertainty);
                            current_source = 'exist';
                        else
                            M = random(task_bayes.gmm);
                            current_source = 'bayes';
                        end
                        M_buffer = [M_buffer; M];
                        source_type_buffer = [source_type_buffer; {current_source}];
                    end
                end
            end
            
            if ~isempty(M_buffer)
                T = elite_sol + M_buffer; 
                T = min(max(T, 0), 1); 
                real_sol = task(n_target).Lb + T .* (task(n_target).Ub - task(n_target).Lb);
                
                trans_fitness = zeros(size(real_sol, 1), 1);
                for k = 1:size(real_sol, 1)
                    trans_fitness(k) = task(n_target).Fnc(real_sol(k, :));
                end
                
                target_pop = task(n_target).population_child;
                replace_num = min(size(real_sol, 1), size(target_pop, 1));
                replace_indices = randperm(size(target_pop, 1), replace_num);
                
                task(n_target).population_child(replace_indices, :) = real_sol(1:replace_num, :);
                
                task_bayes.M_history{gen+1} = M_buffer;
                task_bayes.trans_fitness{gen+1} = trans_fitness;
                task_bayes.M_source{gen+1} = source_type_buffer;
            end
        end
        bayes.(['task' num2str(n_target)]) = task_bayes;
    end
end