import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datasets.bases import read_image
from utils.reranking import re_ranking
from temp import calculate_final_score

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


# def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, match_scores=None):
#     """Evaluation with market1501 metric
#         Key: for each query identity, its gallery images from the same camera view are discarded.
#         """
#     num_q, num_g = distmat.shape
#     # distmat g
#     #    q    1 3 2 4
#     #         4 1 2 3
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     #  0 2 1 3
#     #  1 2 3 0
#     # print('indices: ', indices)
#     # print('g_pids[indices]: ', g_pids[indices])
#     # print('q_pids: ', q_pids)
#     # print('q_pids[indices]: ', q_pids[:, np.newaxis])
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     # print('g_pids[indices] == q_pids[:, np.newaxis]: ', g_pids[indices] == q_pids[:, np.newaxis])
#     # print('matches: ', matches)

#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     num_valid_q = 0.  # number of valid query
#     # print('indices: ', indices)
#     match_dict = {}  

#     # query_images_folder = "/content/drive/MyDrive/SoldierIUST/SOLIDER-REID/datasets/IUST/query"
#     # query_image_names = [img_name for img_name in os.listdir(query_images_folder)]  # Replace [...] with the actual list of query image names

#     # # Path to the folder containing gallery images
#     # gallery_images_folder = "/content/drive/MyDrive/SoldierIUST/SOLIDER-REID/datasets/IUST/bounding_box_test"
#     # gallery_image_names = [img_name for img_name in os.listdir(gallery_images_folder)]
#     # print('distmat: ', distmat)
#     # print(g_pids)

#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         # ##############Check###############
#         # query_image_name = query_image_names[q_idx]
#         # if query_image_name in gallery_image_names:
#         #     print(f"The query image '{query_image_name}' is present in the gallery before removal for Query {q_idx + 1}")
#         # else:
#         #     print(f"The query image '{query_image_name}' is not present in the gallery before removal for Query {q_idx + 1}")
#         # ##################################

#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]  # select one row
#         remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#         keep = np.invert(remove)

#         # ###############Check###############
#         # if query_image_name in [gallery_image_names[order[i]] for i in range(len(order)) if keep[i]]:
#         #     print(f"The query image '{query_image_name}' is still present in the gallery after removal for Query {q_idx + 1}")
#         # else:
#         #     print(f"The query image '{query_image_name}' is not present in the gallery after removal for Query {q_idx + 1}")
#         # ###################################

#         # compute cmc curve
#         # binary vector, positions with value 1 are correct matches
#         orig_cmc = matches[q_idx][keep]
#         # print('orig_cmc',orig_cmc)
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue

#         cmc = orig_cmc.cumsum()
#         # print('cmc', cmc)
#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         # print('num_rel: ', num_rel)
#         tmp_cmc = orig_cmc.cumsum()
#         # print('tmp_cmc: ', tmp_cmc)
#         y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
#         tmp_cmc = tmp_cmc / y
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)
#         # print('AP: ', AP)
#         q_pid = q_pids[q_idx] 
#         q_str_id = str(q_pid).zfill(4)
#         match_dict[q_str_id] = {}
#         # min_dist_scores = np.min(distmat[q_idx, order][keep])
#         for idx, gid in enumerate(order):
#             if not keep[idx]:
#                 continue
#             g_str_id = str(gid).zfill(4)  
#             match_dict[q_str_id][g_str_id] = 1 - distmat[q_idx, gid]


#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)
#     if match_scores is not None:
#         match_scores.update(match_dict)
#     else:
#         match_scores = match_dict

#     # print('match_scores: ', match_scores)

#     return all_cmc, mAP, match_scores

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, match_scores=None):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    # Initialize a dictionary to store the average scores for each query-gallery pair
    match_dict = {}
    
    # Iterate over unique gallery identities
    unique_g_pids = np.unique(g_pids)
    for g_pid in unique_g_pids:
        # Find indices of gallery images belonging to the current identity
        gallery_indices = np.where(g_pids == g_pid)[0]
        # Calculate the average distance scores for all gallery images of the current identity
        avg_score = np.min(distmat[:, gallery_indices], axis=1)
        # Store the average scores in the match_dict dictionary with appropriate keys
        for q_idx, score in enumerate(avg_score):
            q_str_id = str(q_pids[q_idx]).zfill(4)
            g_str_id = str(g_pid).zfill(4)
            if q_str_id not in match_dict:
                match_dict[q_str_id] = {}
            match_dict[q_str_id][g_str_id] = 1 - score
    
    # Update the number of galleries after averaging
    num_g = len(unique_g_pids)

    # Initialize lists to store evaluation metrics
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Compute cmc curve and average precision
        orig_cmc = (g_pids == q_pid).astype(np.int32)
        if not np.any(orig_cmc):
            # Skip if query identity does not appear in the gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # Compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    if match_scores is not None:
        match_scores.update(match_dict)
    else:
        match_scores = match_dict

    return all_cmc, mAP, match_scores


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.image_paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, image_path = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.image_paths.extend(np.asarray(image_path))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = np.asarray(self.image_paths[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = np.asarray(self.image_paths[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            # print('distmat: ', distmat, distmat.shape)
            # print('qf: ', qf, 'gf: ', gf)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        print(qf.shape, gf.shape, distmat.shape)
        cmc, mAP, match_scores = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        result = calculate_final_score(match_scores)

        ## print detials
        indices = np.argsort(distmat, axis=1)
        # query_images_folder = "/content/drive/MyDrive/SoldierIUST/SOLIDER-REID/datasets/IUST/query"
        # query_image_names = [img_name for img_name in os.listdir(query_images_folder)] 

        # Path to the folder containing gallery images
        # gallery_images_folder = "/content/drive/MyDrive/SoldierIUST/SOLIDER-REID/datasets/IUST/bounding_box_test"
        # gallery_image_names = [img_name for img_name in os.listdir(gallery_images_folder)]

        # Print details of top 10 ranked images based on the provided output log

        # parent_folder = "/content/drive/MyDrive/SoldierIUST/SOLIDER-REID/Stats"

        # for i in range(self.num_query):

        #     query_folder = os.path.join(parent_folder, f"Query{i + 1}")
        #     os.makedirs(query_folder, exist_ok=True)

        #     query_img_name = os.path.basename(q_img_paths[i])
        #     query_img_path = os.path.join(query_folder, f"query_{query_img_name}")
        #     query_img = read_image(q_img_paths[i])
        #     query_img.save(query_img_path)

        #     print(f"\nQuery {i + 1}:")
        #     print(f"Query Details - PID: {q_pids[i]}, CamID: {q_camids[i] + 1}, Image Path: {q_img_paths[i]}")
        #     print("Top 10 Ranked Images:")
        #     for j in range(10): 
        #         rank_idx = j  # Use the original rank index directly
        #         idx = indices[i, rank_idx]

        #         gallery_img_name = os.path.basename(g_img_paths[idx])
        #         gallery_img_path = os.path.join(query_folder, f"Rank{rank_idx + 1}_{gallery_img_name}")
        #         gallery_img = read_image(g_img_paths[idx])
        #         gallery_img.save(gallery_img_path)

        #         print(f"Rank {rank_idx + 1} - Similarity Score: {distmat[i, idx]:.4f}, "
        #               f"PID: {g_pids[idx]}, CamID: {g_camids[idx] + 1}, Image Path: {g_img_paths[idx]}")
        # print('q_camids: ', q_camids)  
        # print('g_camids: ', g_camids)            
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



