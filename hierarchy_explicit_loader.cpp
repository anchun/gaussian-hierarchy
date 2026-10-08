/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */


#include "hierarchy_loader.h"
#include "hierarchy_explicit_loader.h"
#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>
#include "half.hpp"
#include "AvgMerger.h"
#include <unordered_set>

Gaussian getRepresentativeGaussian(
	const ExplicitTreeNode* node,
	const std::vector<Gaussian>& gaussians)
{
	if (!node->merged.empty())
		return node->merged.front();
	if (!node->leaf_indices.empty())
		return gaussians[node->leaf_indices.front()];
	for (const ExplicitTreeNode* child : node->children)
	{
		if (!child->merged.empty() || !child->leaf_indices.empty() || !child->children.empty())
			return getRepresentativeGaussian(child, gaussians);
	}
	throw std::runtime_error("Hierarchy contains no representative Gaussian");
}

float getWeight(Eigen::Vector3f& pos, int chunk_id,
	std::vector<Eigen::Vector3f>& chunk_centers)
{
	float dist_to_current_center = (pos - chunk_centers[chunk_id]).norm();
	float min_dist_to_other_center = 1e12f;
	for (int other_chunk_id(0); other_chunk_id < chunk_centers.size(); other_chunk_id++)
	{
		if (other_chunk_id != chunk_id)
		{
			float dist_to_other_center = (pos - chunk_centers[other_chunk_id]).norm();
			if (min_dist_to_other_center > dist_to_other_center)
			{
				min_dist_to_other_center = dist_to_other_center;
			}
		}
	}
	float falloff = 0.05f;
	if (dist_to_current_center <= (1.f - falloff) * min_dist_to_other_center)
	{
		return 1.f;
	}
	else if (dist_to_current_center > (1.f + falloff) * min_dist_to_other_center)
	{
		return 0.f;
	}
	else
	{
		float a = -1.f / (2 * falloff * min_dist_to_other_center);
		float b = (1.f + falloff) / (2 * falloff);
		return a * dist_to_current_center + b;
	}
}

std::vector<ExplicitTreeNode*> buildTreeRec(ExplicitTreeNode* expliciteNode,
	std::vector<Gaussian>& gaussians,
	int chunk_id,
	std::vector<Eigen::Vector3f>& chunk_centers,
	Node& node,
	int node_id,
	std::vector<Eigen::Vector3f>& pos,
	std::vector<SHs>& shs,
	std::vector<float>& alphas,
	std::vector<Eigen::Vector3f>& scales,
	std::vector<Eigen::Vector4f>& rot,
	std::vector<Node>& nodes,
	std::vector<Box>& boxes,
	const std::unordered_set<int>* excluded_nodes,
	bool route_ownership)
{
	if (excluded_nodes != nullptr && excluded_nodes->count(node_id) != 0)
		return {};
	expliciteNode->depth = node.depth;
	expliciteNode->bounds = boxes[node_id];
	int n_valid_gaussians = 0;
	if (node.depth > 0 && !route_ownership)
	{
		for (int n(0); n < node.count_merged; n++)
		{
			float weigth = getWeight(pos[node.start + n], chunk_id, chunk_centers);
			if (weigth > 0.f)
			{
				n_valid_gaussians++;
				Gaussian g;
				g.position = pos[node.start + n];
				g.rotation = rot[node.start + n];
				g.opacity = alphas[node.start + n] * weigth;
				g.scale = scales[node.start + n].array().exp();
				g.shs = shs[node.start + n];

				expliciteNode->merged.push_back(g);
			}
		}
	}
	else
	{
		for (int n(0); n < node.count_leafs; n++)
		{
			float weigth = route_ownership ? 1.f : getWeight(pos[node.start + n], chunk_id, chunk_centers);
			if (weigth > 0.f)
			{
				n_valid_gaussians++;
				Gaussian g;
				g.position = pos[node.start + n];
				g.rotation = rot[node.start + n];
				g.opacity = alphas[node.start + n] * weigth;
				g.scale = scales[node.start + n].array().exp();
				g.shs = shs[node.start + n];

				expliciteNode->leaf_indices.push_back(gaussians.size());
				gaussians.push_back(g);
			}
		}
	}

	std::vector<ExplicitTreeNode*> children;
	for (int i = 0; i < node.count_children; i++)
	{
		ExplicitTreeNode* newNode = new ExplicitTreeNode;
		std::vector<ExplicitTreeNode*> newChildren = buildTreeRec(newNode, gaussians, chunk_id, chunk_centers,
			nodes[node.start_children + i], node.start_children + i,
			pos, shs, alphas, scales, rot, nodes, boxes, excluded_nodes, route_ownership
		);
		children.insert(children.end(), newChildren.begin(), newChildren.end());
	}
	if (route_ownership && node.depth > 0)
	{
		if (children.empty())
			return {};
		expliciteNode->children = children;
		std::vector<Gaussian> representatives;
		representatives.reserve(children.size());
		for (const ExplicitTreeNode* child : children)
			representatives.push_back(getRepresentativeGaussian(child, gaussians));
		expliciteNode->merged.push_back(AvgMerger::mergeGaussians(representatives));
		return {expliciteNode};
	}

	if (n_valid_gaussians > 0)
	{
		expliciteNode->children = children;
		return std::vector<ExplicitTreeNode*>(1, expliciteNode);
	}
	else
	{
		return children;
	}
}

void HierarchyExplicitLoader::loadExplicit(
	const char* filename, std::vector<Gaussian>& gaussians, ExplicitTreeNode* root,
	int chunk_id, std::vector<Eigen::Vector3f>& chunk_centers,
	const std::unordered_set<int>* excluded_nodes,
	bool route_ownership)
{
	std::vector<Eigen::Vector3f> pos;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Eigen::Vector3f> scales;
	std::vector<Eigen::Vector4f> rot;
	std::vector<Node> nodes;
	std::vector<Box> boxes;
	HierarchyLoader::load(filename, pos, shs, alphas, scales, rot, nodes, boxes);

	pos[0] = chunk_centers[chunk_id];
	std::vector<ExplicitTreeNode*> loaded_roots = buildTreeRec(
		root, gaussians, chunk_id, chunk_centers,
		nodes[0], 0, pos, shs, alphas, scales, rot, nodes, boxes,
		excluded_nodes, route_ownership);
	if (loaded_roots.size() == 1 && loaded_roots.front() == root)
		return;
	if (loaded_roots.empty())
		throw std::runtime_error("Hierarchy has no Gaussians after applying merge weights");

	root->children = loaded_roots;
	std::vector<Gaussian> representatives;
	representatives.reserve(loaded_roots.size());
	for (const ExplicitTreeNode* loaded_root : loaded_roots)
		representatives.push_back(getRepresentativeGaussian(loaded_root, gaussians));
	root->merged.push_back(AvgMerger::mergeGaussians(representatives));
}
