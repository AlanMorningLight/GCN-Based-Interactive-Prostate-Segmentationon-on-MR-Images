import torch
import torch.nn.functional as F
import numpy as np
import Utils.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """

    assert(output.size() == label.size())

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_pos_pix = torch.mul(w_void, loss_pos_pix)
        loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = torch.sum(loss_pos_pix)
    loss_neg = torch.sum(loss_neg_pix)

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss

def fp_edge_loss(gt_edges, edge_logits):
    """
    Edge loss in the first point network

    gt_edges: [batch_size, grid_size, grid_size] of 0/1
    edge_logits: [batch_size, grid_size*grid_size]
    """
    edges_shape = gt_edges.size()
    gt_edges = gt_edges.view(edges_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges)

    return torch.mean(loss)

def fp_vertex_loss(gt_verts, vertex_logits):
    """
    Vertex loss in the first point network
    
    gt_verts: [batch_size, grid_size, grid_size] of 0/1
    vertex_logits: [batch_size, grid_size**2]
    """
    verts_shape = gt_verts.size()
    gt_verts = gt_verts.view(verts_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(vertex_logits, gt_verts)

    return torch.mean(loss)



def poly_mathcing_loss(pnum, pred, gt, loss_type="L2"):
    '''
    :param pnum: the number of spline vertices
    :param pred: [bs, pnum, 2 ] \in (0,1)
    :param gt: [bs, pnum, 2 ] \in (0,1)
    :param loss_type:
    :return:
    '''

    batch_size = pred.size()[0]
    pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
    for b in range(batch_size):
        for i in range(pnum):
            pidx = (np.arange(pnum) + i) % pnum
            pidxall[b, i] = pidx

    pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

    # import ipdb;
    # ipdb.set_trace()
    feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), gt.size(2)).detach()
    gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)

    pred_expand = pred.unsqueeze(1)

    dis = pred_expand - gt_expand   ##return the distance from pred_expand to gt_expand


    if loss_type == "L2":
        dis = (dis ** 2).sum(3).sqrt().sum(2)
    elif loss_type == "L1":
        dis = torch.abs(dis).sum(3).sum(2)

    min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
    min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)

    min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
                            expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
    gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

    return gt_right_order, torch.mean(min_dis)

def poly_match_interactive(pnum, pred, gt, loss_type="L2"):
    '''
    :param pnum: the number of spline vertices
    :param pred: [bs, pnum, 2 ] \in (0,1)
    :param gt: [bs, pnum, 2 ] \in (0,1)
    :param loss_type:
    :return:
    '''

    batch_size = pred.size()[0]
    pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
    for b in range(batch_size):
        for i in range(pnum):
            pidx = (np.arange(pnum) + i) % pnum
            pidxall[b, i] = pidx

    pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

    # import ipdb;
    # ipdb.set_trace()
    feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), gt.size(2)).detach() #(bs, pnum*pnum, 2)
    gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2) #(bs, pnum,pnum,2)

    pred_expand = pred.unsqueeze(1) #(bs,1,pnum,2)

    dis = pred_expand - gt_expand  # [bs,pnum,pnum,2] ##return the distance from pred_expand to gt_expand


    if loss_type == "L2":
        dis = (dis ** 2).sum(3).sqrt().sum(2)
    elif loss_type == "L1":
        ## note: bellow code calc the sum of each match, and calc the sum of match
        # pre is stable, gt is looped
        dis_match = torch.abs(dis).sum(3) # (2, 40, 40)
        dis = torch.abs(dis).sum(3).sum(2) # (bs,40)

    min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
    shortest_match = torch.zeros(gt.shape[0], 1, pnum)
    # shortest_match = dis_match[:,min_id,:]
    for i in range(gt.shape[0]):
        shortest_match[i,0,:] = dis_match[i,min_id[i,0], :] # (bs,1, pnum)
    final_max_dis, final_max_dis_id = torch.max(shortest_match, dim=2) # find the "minmax" distance, final_max_dis:(bs,1), final_max_dis_id:(bs,1)
    final_match = torch.zeros(gt.shape[0], 1, pnum, 2)
    for i in range(gt.shape[0]):
        final_match[i,0,:,:]  = gt_expand[i,min_id[i,0],:,:]


    return final_match, final_max_dis_id   #final_match: (bs, 1, pnum, 2) ; final_max_dis_id: (bs, 1)

def GeneralizedDice(probs, onehot_labels):
    '''
    :param probs: b2wh, the probs of last CNN layer input to F.log_softmax()
    :param onehot_labels: one-hot operation labels
    :return:
    '''
    #assert utils.checkSimplex_(probs) and utils.checkSimplex_(onehot_labels)
    idc = [0, 1]
    #pc = probs[:, idc, ...].type(torch.float32) #pc: bcwh
    pc = probs.type(torch.float32)
    #tc = onehot_labels[:, idc, ...].type(torch.float32)
    tc = onehot_labels #convert ndarray to Tensor
    pc_ = torch.zeros(pc.shape[0], pc.shape[1]).cuda()   # bc
    tc_ = torch.zeros(tc.shape[0], tc.shape[1]).cuda() #bc
    ## intersection = torch.einsum('bcwh, bcwh -> bc', [pc, tc])
    temp = F.mul(tc, pc)    #bcwh
    intersection_ = torch.zeros(pc.shape[0], pc.shape[1])
    ## bellow operation equals as
    ## pc_= torch.einsum('bcwh', [pc])  tc_ = torch.einsum('bcwh -> bc', [tc])
    for vi in range(pc.shape[0]):
        for vj in range(pc.shape[1]):
            pc_[vi][vj] = torch.sum(pc[vi,vj,...])
            tc_[vi][vj] = torch.sum(tc[vi,vj,:,:])
            intersection_ = torch.sum(temp[vi, vj,:,:])    #intersection of pre mask and GT mask
    w = 1 / ((pc_.type(torch.float32) + 1e-10) ** 2)
    intersection = w * intersection_
    union = w * (pc_ + tc_) # union of pre mask and GT mask

    divided = 1-2 * (torch.sum(intersection, 1) + 1e-10) / (torch.sum(union, 1) + 1e-10)
    #loss = divided.mean()
    loss = torch.mean(divided)
    return loss


def SurfaceLoss(probs, dis_map):
    #assert utils.checkSimplex_(probs)
    assert not utils.one_hot(dis_map)  #if false throw exception e
    idc = [1]
    #pc = probs[:, idc, ...].type(torch.float32) #bcwh
    pc = probs[:,1,:].type(torch.float32)
    dc = dis_map[:, 1, ...] #bcwh

    multipled = F.mul(pc, dc)   #bcwh, equal to 'torch.einsum('bcwh, bcwh -> bcwh', [pc, dc])'
    #loss = multipled.mean()
    loss = torch.mean(multipled)
    return loss