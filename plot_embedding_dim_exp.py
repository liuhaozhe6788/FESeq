from fuxictr.utils import save_embedding_dim_exp_results

auc = [[[8,16,24,32],[0.611757,0.612739,0.606080,0.605284]],  # eleme
       [[8,16,24,32],[0.876226,0.891220,0.884504,0.884520]]]  # bund;e

logloss = [[[8,16,24,32],[0.092231,0.091790,0.092195,0.092225]],
        [[8,16,24,32],[0.012033,0.011678,0.011026,0.011334]]]

save_embedding_dim_exp_results(auc, logloss, ylim11=[0.56, 0.64], ylim12=[0.8, 0.9], ylim21=[0.083, 0.095], ylim22=[0.005, 0.015])