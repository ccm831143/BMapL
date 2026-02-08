% 迁移结果评估
function [scores, metrics] = Feedback_assessment(target_pop, trans_fitness, target_fitness)
    % 参数设置
    diversity_weight = 0.2;
    contribution_weight = 0.5;
    improvement_weight = 0.3;
    
    % 初始化指标
    n_trans = length(trans_fitness);
    diversity_contrib = zeros(n_trans,1);
    contribution = zeros(n_trans,1);
    improvement = zeros(n_trans,1);
    
    %% 1. 计算原始指标值 ------------------------------------------------
    % 多样性贡献：解到目标种群的平均距离
    for i = 1:n_trans
        trans_sol = target_pop(end-n_trans+i,:); 
        dists = pdist2(trans_sol, target_pop(1:end-n_trans,:));
        diversity_contrib(i) = mean(dists);
    end
    
    % 前沿贡献度：在合并种群中的归一化排名
    combined_fitness = [target_fitness; trans_fitness];
    [~, sorted_idx] = sort(combined_fitness);
    ranks = zeros(size(combined_fitness));
    ranks(sorted_idx) = 1:length(combined_fitness);
    trans_ranks = ranks(length(target_fitness)+1:end);
    contribution = 1 - (trans_ranks - 1) / length(combined_fitness);
    
    % 改进率：处理负值并计算原始值
    prev_best = min(target_fitness);
    raw_improvement = (prev_best - trans_fitness) / prev_best;
    improvement = max(raw_improvement, 0); % 负改进率归零
    
    %% 2. 指标规范化 ----------------------------------------------------
    % 多样性贡献：min-max归一化到[0,1]
    div_min = min(diversity_contrib);
    div_max = max(diversity_contrib);
    if div_max > div_min
        diversity_norm = (diversity_contrib - div_min) / (div_max - div_min);
    else
        diversity_norm = zeros(size(diversity_contrib));
    end
    
    % 改进率：min-max归一化到[0,1]
    imp_min = min(improvement);
    imp_max = max(improvement);
    if imp_max > imp_min
        improvement_norm = (improvement - imp_min) / (imp_max - imp_min);
    else
        improvement_norm = zeros(size(improvement));
    end
    
    % 前沿贡献度：已在[0,1]，直接使用
    contribution_norm = contribution;
    
    %% 3. 综合评分计算 --------------------------------------------------
    scores = diversity_weight * diversity_norm + ...
             contribution_weight * contribution_norm + ...
             improvement_weight * improvement_norm;
    
    %% 4. 返回规范化后的指标-----------------------------------
    metrics.diversity = mean(diversity_norm);
    metrics.contribution = mean(contribution_norm);
    metrics.improvement = mean(improvement_norm);
end