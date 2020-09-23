import math
import torch
import torch.nn.functional as F
from .. import utils

class ObjectEncoder(object):

    def __init__(self, classnames=['Car'], pos_std=[.5, .36, .5], 
                 log_dim_mean=[[0.42, 0.48, 1.35]], 
                 log_dim_std=[[.085, .067, .115]], sigma=1., nms_thresh=0.05):
        
        self.classnames = classnames
        self.nclass = len(classnames)
        self.pos_std = torch.tensor(pos_std)
        self.log_dim_mean = torch.tensor(log_dim_mean)
        self.log_dim_std = torch.tensor(log_dim_std)

        self.sigma = sigma
        self.nms_thresh = nms_thresh
        
    
    def encode_batch(self, objects, grids):

        # Encode batch element by element
        batch_encoded = [self.encode(objs, grid) for objs, grid 
                         in zip(objects, grids)]
        
        # Transpose batch
        # labels, pos_offsets, dim_offsets, ang_offsets = zip(*batch_encoded)
        # return torch.stack(labels), torch.stack(pos_offsets), \
        #     torch.stack(dim_offsets), torch.stack(ang_offsets)
        labels, sqr_dists, pos_offsets, dim_offsets, ang_offsets = zip(*batch_encoded)
        return torch.stack(labels), torch.stack(sqr_dists), torch.stack(pos_offsets), \
            torch.stack(dim_offsets), torch.stack(ang_offsets)



    def encode(self, objects, grid):
        
        # Filter objects by class name
        objects = [obj for obj in objects if obj.classname in self.classnames]

        # Skip empty examples
        if len(objects) == 0:
            return self._encode_empty(grid)
        
        # Construct tensor representation of objects
        classids = torch.tensor([self.classnames.index(obj.classname) 
                                for obj in objects], device=grid.device)
        positions = grid.new([obj.position for obj in objects])
        dimensions = grid.new([obj.dimensions for obj in objects])
        angles = grid.new([obj.angle for obj in objects])

        # Assign objects to locations on the grid
        labels, indices = self._assign_to_grid(
            classids, positions, dimensions, angles, grid)

        sqr_dists = self._encode_distances(classids, positions, grid)
        
        # Encode positions, dimensions and angles
        pos_offsets = self._encode_positions(positions, indices, grid)
        dim_offsets = self._encode_dimensions(classids, dimensions, indices)
        ang_offsets = self._encode_angles(angles, indices)

        return labels, sqr_dists, pos_offsets, dim_offsets, ang_offsets   
    

    def _assign_to_grid(self, classids, positions, dimensions, angles, grid):

        # Compute grid centers
        centers = (grid[1:, 1:, :] + grid[:-1, :-1, :]) / 2.

        # Transform grid into object coordinate systems
        local_grid = utils.rotate(centers - positions.view(-1, 1, 1, 3), 
            -angles.view(-1, 1, 1)) / dimensions.view(-1, 1, 1, 3)
        
        # Find all grid cells which lie within each object
        inside = (local_grid[..., [0, 2]].abs() <= 0.5).all(dim=-1)
        
        # Expand the mask in the class dimension NxDxW * NxC => NxCxDxW
        class_mask = classids.view(-1, 1) == torch.arange(
            len(self.classnames)).type_as(classids)
        class_inside = inside.unsqueeze(1) & class_mask[:, :, None, None]

        # Return positive locations and the id of the corresponding instance 
        labels, indices = torch.max(class_inside, dim=0)
        return labels, indices
    
    def _encode_distances(self, classids, positions, grid):

        centers = (grid[1:, 1:, [0, 2]] + grid[:-1, :-1, [0, 2]]) / 2.
        positions = positions.view(-1, 1, 1, 3)[..., [0, 2]]

        # Compute squared distances
        obj_sqr_dists = (positions - centers).pow(2).sum(dim=-1)
        sqr_dists, _ = obj_sqr_dists.min(dim=0)

        # Normalize by grid resolution
        sigma0 = (grid[1:, 1:, [0, 2]] - grid[:-1, :-1, [0, 2]] / 2.).pow(2).sum(-1) / 4.
        sqr_dists = sqr_dists / sigma0

        # TODO handle classes correctly
        return sqr_dists.unsqueeze(0)

    

    def _encode_positions(self, positions, indices, grid):

        # Compute the center of each grid cell
        centers = (grid[1:, 1:] + grid[:-1, :-1]) / 2.

        # Encode positions into the grid
        C, D, W = indices.size()
        positions = positions.index_select(0, indices.view(-1)).view(C, D, W, 3)

        # Compute relative offsets and normalize
        pos_offsets = (positions - centers) / self.pos_std
        return pos_offsets.permute(0, 3, 1, 2)
    
    def _encode_dimensions(self, classids, dimensions, indices):
        
        # Convert mean and std to tensors
        log_dim_mean = self.log_dim_mean[classids]
        log_dim_std = self.log_dim_std[classids]

        # Compute normalized log scale offset
        dim_offsets = (torch.log(dimensions) - log_dim_mean) / log_dim_std

        # Encode dimensions to grid
        C, D, W = indices.size()
        dim_offsets = dim_offsets.index_select(0, indices.view(-1))
        return dim_offsets.view(C, D, W, 3).permute(0, 3, 1, 2)
    

    def _encode_angles(self, angles, indices):

        # Compute rotation vector
        sin = torch.sin(angles)[indices]
        cos = torch.cos(angles)[indices]
        return torch.stack([cos, sin], dim=1)
    

    def _encode_empty(self, grid):
        depth, width, _ = grid.size()
        max_dist = math.sqrt(depth ** 2 + width ** 2)

        # Generate empty tensors
        labels = grid.new_zeros((self.nclass, depth-1, width-1)).byte()
        sqr_dists = grid.new_full((self.nclass, depth-1, width-1), max_dist)
        pos_offsets = grid.new_zeros((self.nclass, 3, depth-1, width-1))
        dim_offsets = grid.new_zeros((self.nclass, 3, depth-1, width-1))
        ang_offsets = grid.new_zeros((self.nclass, 2, depth-1, width-1))
        
        return labels, sqr_dists, pos_offsets, dim_offsets, ang_offsets
        #return heatmaps, pos_offsets, dim_offsets, ang_offsets, mask
    

    def decode(self, heatmaps, pos_offsets, dim_offsets, ang_offsets, grid):
        
        # Apply NMS to find positive heatmap locations
        peaks, scores, classids = self._decode_heatmaps(heatmaps)

        # Decode positions, dimensions and rotations
        positions = self._decode_positions(pos_offsets, peaks, grid)
        dimensions = self._decode_dimensions(dim_offsets, peaks)
        angles = self._decode_angles(ang_offsets, peaks)

        objects = list()
        for score, cid, pos, dim, ang in zip(scores, classids, positions, 
                                             dimensions, angles):
            objects.append(utils.ObjectData(
                self.classnames[cid], pos, dim, ang, score))
        
        return objects

    def _decode_heatmaps(self, heatmaps):
        peaks = non_maximum_suppression(heatmaps, self.sigma)
        scores = heatmaps[peaks]
        classids = torch.nonzero(peaks)[:, 0]
        return peaks, scores, classids


    def _decode_positions(self, pos_offsets, peaks, grid):

        # Compute the center of each grid cell
        centers = (grid[1:, 1:] + grid[:-1, :-1]) / 2.

        # Un-normalize grid offsets
        positions = pos_offsets.permute(0, 2, 3, 1) * self.pos_std + centers
        return positions[peaks]
    
    def _decode_dimensions(self, dim_offsets, peaks):
        dim_offsets = dim_offsets.permute(0, 2, 3, 1)
        dimensions = torch.exp(
            dim_offsets * self.log_dim_std + self.log_dim_mean)
        return dimensions[peaks]
    
    def _decode_angles(self, angle_offsets, peaks):
        cos, sin = torch.unbind(angle_offsets, 1)
        return torch.atan2(sin, cos)[peaks]



def non_maximum_suppression(heatmaps, sigma=1.0, thresh=0.05, max_peaks=50):
    
    # Smooth with a Gaussian kernel
    num_class = heatmaps.size(0)
    kernel = utils.gaussian_kernel(sigma).to(heatmaps)
    kernel = kernel.expand(num_class, num_class, -1, -1)
    smoothed = F.conv2d(
        heatmaps[None], kernel, padding=int((kernel.size(2)-1)/2))

    # Max pool over the heatmaps
    max_inds = F.max_pool2d(smoothed, 3, stride=1, padding=1, 
                               return_indices=True)[1].squeeze(0)

    # Find the pixels which correspond to the maximum indices
    _, height, width = heatmaps.size()
    flat_inds = torch.arange(height*width).type_as(max_inds).view(height, width)
    peaks = (flat_inds == max_inds) & (heatmaps > thresh)
    
    # Keep only the top N peaks
    if peaks.long().sum() > max_peaks:
        scores = heatmaps[peaks]
        scores, _ = torch.sort(scores, descending=True)
        peaks = peaks & (heatmaps > scores[max_peaks-1])
    
    return peaks