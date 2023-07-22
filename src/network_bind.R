library(igraph)
library(tnet)

rm(list=ls())
# setwd("/Users/wangyuchen/Desktop/VSCode/summer/DeepNetBim-master/src")

set.seed(1234)

mhc = read.csv('../data/binding_origin.csv')

mhc <- mhc[,-4]
mhc <- mhc[, c(1, 3, 2)]
# mhc <- head(mhc,20)
# mhc$test <- 1-(log(mhc$affinity_raw))/log(20000)

# 将affinity列中小于等于0的值替换为10^(-10)，确保affinity为正数
mhc$affinity_ = ifelse(mhc$affinity<=0, 10^(-10), mhc$affinity)

# 创建一个无向图g，并为其添加边属性"weight"，其值来自mhc数据框中的affinity_列
g <- graph.data.frame(mhc, directed = FALSE) %>% set_edge_attr('weight', value = as.numeric(mhc$affinity_))

# plot(g, layout = layout_with_fr(g), edge.label = E(g)$weight)

# 为图中的节点V(g)设置类型（type），使用bipartite_mapping函数进行划分
V(g)$type <- bipartite_mapping(g)$type

# 计算图中所有节点的度（degree），并进行标准化处理
all_degree <- degree(g, normalized = TRUE)

# 计算图中所有节点的紧密度（closeness）
all_close <- closeness(g)

# 计算图中所有节点的介数中心性（betweenness）
all_between <- betweenness(g)

# 计算图中所有节点的特征向量中心性（eigenvector centrality），并且不进行标准化
all_evcent <- evcent(g, scale = FALSE)$vector

# 将计算得到的节点特征合并为一个数据框all_feature
all_feature <- cbind(all_degree, all_close, all_between, all_evcent)
all_feature <- as.data.frame(all_feature)

# 提取所有HLA相关的特征，存储在hla_feature数据框中，并添加一个名为mhc的列
hla_feature <- all_feature[grep('HLA', rownames(all_feature)), ]
hla_feature$mhc <- rownames(hla_feature)
colnames(hla_feature) <- gsub('all', 'hla', colnames(hla_feature))

# 提取除HLA外的肽段相关特征，存储在pep_feature数据框中，并添加一个名为pep_encode的列
pep_feature <- all_feature[-grep('HLA', rownames(all_feature)), ]
pep_feature$sequence <- rownames(pep_feature)
colnames(pep_feature) <- gsub('all', 'pep', colnames(pep_feature))

# 将mhc和pep特征合并为一个新的数据框tmp
tmp <- merge(mhc, hla_feature, by = 'mhc')
tmp <- merge(tmp, pep_feature, by = 'sequence')

# 对tmp数据框的第9列到第16列进行标准化处理
for (i in 5:12){
  tmp[, i] = scale(tmp[, i])
}

# tmp <- tmp[!grepl(",", tmp$mhc), ]

# 将处理后的数据保存为bind_train.csv文件
write.csv(tmp, file = '../data/bind_all.csv',row.names = FALSE)
