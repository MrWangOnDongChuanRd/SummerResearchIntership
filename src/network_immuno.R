library(dplyr)
library(igraph)
library(tnet)

rm(list=ls())

# setwd("/Users/wangyuchen/Desktop/VSCode/summer/DeepNetBim-master/src")

mhc <- read.csv('../data/immunogenic_origin.csv')
# mhc <- head(mhc,20)

# 使用 filter() 函数来筛选数据
filtered_mhc <- mhc %>%
  filter(nchar(sequence) == 9)

# 将mhc序列编码为因子，用唯一的序列作为水平，用1至唯一序列数作为标签
pep_encode <- factor(mhc$sequence, levels = unique(mhc$sequence), labels = 1:length(unique(mhc$sequence)))

# 将mhc的分子编码为因子，用唯一的分子名称作为水平，用1至唯一分子数作为标签
hla_encode <- factor(mhc$mhc, levels = unique(mhc$mhc), labels = 1:length(unique(mhc$mhc)))

# 将hla_encode和pep_encode作为新的列添加到mhc数据框中
mhc$hla_encode <- hla_encode
mhc$pep_encode <- pep_encode

mhc <- mhc[c('mhc', 'pep_encode', 'hla_encode', 'sequence', 'Label')]

# 将mhc数据框中Label列中的0值替换为10^(-10)
mhc$Label[which(mhc$Label == 0)] = 10^(-10)

# 根据mhc数据框创建一个图，未指定有向图(directed = F)
g <- graph.data.frame(mhc, directed = F) %>%
  # 将边的属性'weight'设置为Label列的数值，并转换为数值类型
  set_edge_attr('weight', value = as.numeric(mhc$Label))

# plot(g, layout = layout_with_fr(g), edge.label = E(g)$weight)

# 用bipartite_mapping函数为图中的节点设置'type'属性，用于指示节点的类型（hla或pep）
V(g)$type <- bipartite_mapping(g)$type

# 计算图中所有节点的度（degree）并进行标准化处理
all_degree <- degree(g)

# 计算图中所有节点的接近度（closeness）
all_close <- closeness(g)

# 计算图中所有节点的介数中心性（betweenness）
all_between <- betweenness(g)

# 计算图中所有节点的特征向量中心性（eigenvector centrality），不进行缩放
all_evcent <- evcent(g, scale = F)$vector

# 将计算得到的度、接近度、介数中心性和特征向量中心性合并为一个数据框
all_feature <- cbind(all_degree, all_close, all_between, all_evcent)
all_feature <- as.data.frame(all_feature)

# 根据节点名称中是否包含'HLA'关键字，将特征分为hla_feature和pep_feature
hla_feature <- all_feature[grep('HLA', rownames(all_feature)), ]
hla_feature$mhc <- rownames(hla_feature)
colnames(hla_feature) <- gsub('all', 'hla', colnames(hla_feature))

pep_feature <- all_feature[-grep('HLA', rownames(all_feature)), ]
pep_feature$pep_encode <- rownames(pep_feature)
colnames(pep_feature) <- gsub('all', 'pep', colnames(pep_feature))

# 根据'mhc'和'pep_encode'列将特征数据框与原始mhc数据框合并
tmp <- merge(mhc, hla_feature, by = 'mhc')
tmp <- merge(tmp, pep_feature, by = 'pep_encode')

# 对第6列至第13列的特征数据进行标准化处理
for (i in 6:13) {
  tmp[, i] = scale(tmp[, i])
}

# 将处理后的tmp数据框写入CSV文件'immuno_train.csv'
write.csv(tmp, file = 'immuno_all.csv',row.names = FALSE)
