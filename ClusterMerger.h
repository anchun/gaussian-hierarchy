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

#pragma once

#include "common.h"
#include <functional>
#include <limits>

class ClusterMerger
{
private:
	float max_merge_scale;
	std::function<bool(const Gaussian&)> merge_predicate;
	void mergeRec(ExplicitTreeNode* node, const std::vector<Gaussian>& leaf_gaussians);
public:
	explicit ClusterMerger(
		float max_merge_scale = std::numeric_limits<float>::infinity(),
		std::function<bool(const Gaussian&)> merge_predicate = {});
	void merge(ExplicitTreeNode* root, const std::vector<Gaussian>& gaussians);
};