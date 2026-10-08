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
#include <vector>
#include <Eigen/Dense>
#include "common.h"
#include <iostream>
#include <fstream>
#include "half.hpp"
#include <algorithm>
#include <limits>

struct HalfBox2
{
	half_float::half minn[4];
	half_float::half maxx[4];
};

void HierarchyLoader::load(const char* filename,
	std::vector<Eigen::Vector3f>& pos,
	std::vector<SHs>& shs,
	std::vector<float>& alphas,
	std::vector<Eigen::Vector3f>& scales,
	std::vector<Eigen::Vector4f>& rot,
	std::vector<Node>& nodes,
	std::vector<Box>& boxes)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error("File not found!");

	int P;
	infile.read((char*)&P, sizeof(int));

	if (P >= 0)
	{
		pos.resize(P);
		shs.resize(P);
		alphas.resize(P);
		scales.resize(P);
		rot.resize(P);

		infile.read((char*)pos.data(), P * sizeof(Eigen::Vector3f));
		infile.read((char*)rot.data(), P * sizeof(Eigen::Vector4f));
		infile.read((char*)scales.data(), P * sizeof(Eigen::Vector3f));
		infile.read((char*)alphas.data(), P * sizeof(float));
		infile.read((char*)shs.data(), P * sizeof(SHs));

		int N;
		infile.read((char*)&N, sizeof(int));

		nodes.resize(N);
		boxes.resize(N);

		infile.read((char*)nodes.data(), N * sizeof(Node));
		infile.read((char*)boxes.data(), N * sizeof(Box));
	}
	else
	{
		size_t allP = -P;

		pos.resize(allP);
		infile.read((char*)pos.data(), allP * sizeof(Eigen::Vector3f));
		// lower the memory cost
		{
			rot.resize(allP);
			std::vector<half_float::half> half_rotations(allP * 4);
			infile.read((char*)half_rotations.data(), allP * 4 * sizeof(half_float::half));
			for (size_t i = 0; i < allP; i++)
			{
				for (size_t j = 0; j < 4; j++)
					rot[i][j] = half_rotations[i * 4 + j];
			}
		}
		{
			scales.resize(allP);
			std::vector<half_float::half> half_scales(allP * 3);
			infile.read((char*)half_scales.data(), allP * 3 * sizeof(half_float::half));
			for (size_t i = 0; i < allP; i++)
			{
				for (size_t j = 0; j < 3; j++)
					scales[i][j] = half_scales[i * 3 + j];
			}
		}
		{
			alphas.resize(allP);
			std::vector<half_float::half> half_opacities(allP);
			infile.read((char*)half_opacities.data(), allP * sizeof(half_float::half));
			for (size_t i = 0; i < allP; i++)
			{
				alphas[i] = half_opacities[i];
			}
		}
		{
			shs.resize(allP);
			std::vector<half_float::half> half_shs(allP * 48);
			infile.read((char*)half_shs.data(), allP * 48 * sizeof(half_float::half));
			for (size_t i = 0; i < allP; i++)
			{
				for (size_t j = 0; j < 48; j++)
					shs[i][j] = half_shs[i * 48 + j];
			}
		}

		int N;
		infile.read((char*)&N, sizeof(int));
		size_t allN = N;

		{
			nodes.resize(allN);
			std::vector<HalfNode> half_nodes(allN);
			infile.read((char*)half_nodes.data(), allN * sizeof(HalfNode));
			for (int i = 0; i < allN; i++)
			{
				nodes[i].parent = half_nodes[i].parent;
				nodes[i].start = half_nodes[i].start;
				nodes[i].start_children = half_nodes[i].start_children;
				nodes[i].depth = half_nodes[i].dccc[0];
				nodes[i].count_children = half_nodes[i].dccc[1];
				nodes[i].count_leafs = half_nodes[i].dccc[2];
				nodes[i].count_merged = half_nodes[i].dccc[3];
			}
		}

		{
			boxes.resize(allN);
			std::vector<HalfBox2> half_boxes(allN);
			infile.read((char*)half_boxes.data(), allN * sizeof(HalfBox2));
			for (int i = 0; i < allN; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					boxes[i].minn[j] = half_boxes[i].minn[j];
					boxes[i].maxx[j] = half_boxes[i].maxx[j];
				}
			}
		}
	}
}

bool HierarchyLoader::loadAdaptiveUnits(const char* filename,
	float target_extent,
	std::vector<int>& node_ids,
	std::vector<Box>& unit_boxes,
	int& selected_depth,
	std::string& error,
	int requested_depth)
{
	std::ifstream infile(filename, std::ios_base::binary);
	if (!infile.good())
	{
		error = std::string("cannot open hierarchy ") + filename;
		return false;
	}
	int point_count;
	infile.read(reinterpret_cast<char*>(&point_count), sizeof(int));
	if (!infile.good() || point_count == std::numeric_limits<int>::min())
	{
		error = std::string("invalid hierarchy header in ") + filename;
		return false;
	}
	const bool compressed = point_count < 0;
	const std::size_t gaussian_count = static_cast<std::size_t>(std::abs(point_count));
	const std::size_t gaussian_record_bytes = compressed
		? 3 * sizeof(float) + (4 + 3 + 1 + 48) * sizeof(half_float::half)
		: (3 + 48 + 1 + 3 + 4) * sizeof(float);
	infile.seekg(static_cast<std::streamoff>(gaussian_count * gaussian_record_bytes), std::ios_base::cur);
	int node_count;
	infile.read(reinterpret_cast<char*>(&node_count), sizeof(int));
	if (!infile.good() || node_count <= 0)
	{
		error = std::string("invalid hierarchy topology in ") + filename;
		return false;
	}

	constexpr std::size_t kBlockSize = 1 << 20;
	std::vector<unsigned char> depths(static_cast<std::size_t>(node_count));
	int maximum_depth = 0;
	for (std::size_t offset = 0; offset < static_cast<std::size_t>(node_count); offset += kBlockSize)
	{
		const std::size_t count = std::min(kBlockSize, static_cast<std::size_t>(node_count) - offset);
		if (compressed)
		{
			std::vector<HalfNode> nodes(count);
			infile.read(reinterpret_cast<char*>(nodes.data()), count * sizeof(HalfNode));
			for (std::size_t index = 0; index < count; ++index)
			{
				const int depth = nodes[index].dccc[0];
				if (depth < 0 || depth > 255)
				{
					error = "hierarchy node depth is out of range";
					return false;
				}
				depths[offset + index] = static_cast<unsigned char>(depth);
				maximum_depth = std::max(maximum_depth, depth);
			}
		}
		else
		{
			std::vector<Node> nodes(count);
			infile.read(reinterpret_cast<char*>(nodes.data()), count * sizeof(Node));
			for (std::size_t index = 0; index < count; ++index)
			{
				const int depth = nodes[index].depth;
				if (depth < 0 || depth > 255)
				{
					error = "hierarchy node depth is out of range";
					return false;
				}
				depths[offset + index] = static_cast<unsigned char>(depth);
				maximum_depth = std::max(maximum_depth, depth);
			}
		}
		if (!infile.good())
		{
			error = "failed while reading hierarchy nodes";
			return false;
		}
	}

	const int minimum_candidate_depth = std::max(0, maximum_depth - 20);
	std::vector<std::vector<std::pair<int, Box>>> candidates(static_cast<std::size_t>(maximum_depth + 1));
	for (std::size_t offset = 0; offset < static_cast<std::size_t>(node_count); offset += kBlockSize)
	{
		const std::size_t count = std::min(kBlockSize, static_cast<std::size_t>(node_count) - offset);
		if (compressed)
		{
			std::vector<HalfBox2> boxes(count);
			infile.read(reinterpret_cast<char*>(boxes.data()), count * sizeof(HalfBox2));
			for (std::size_t index = 0; index < count; ++index)
			{
				const int depth = depths[offset + index];
				if (depth < minimum_candidate_depth)
					continue;
				Box box;
				for (int axis = 0; axis < 4; ++axis)
				{
					box.minn[axis] = boxes[index].minn[axis];
					box.maxx[axis] = boxes[index].maxx[axis];
				}
				candidates[depth].emplace_back(static_cast<int>(offset + index), box);
			}
		}
		else
		{
			std::vector<Box> boxes(count);
			infile.read(reinterpret_cast<char*>(boxes.data()), count * sizeof(Box));
			for (std::size_t index = 0; index < count; ++index)
			{
				const int depth = depths[offset + index];
				if (depth >= minimum_candidate_depth)
					candidates[depth].emplace_back(static_cast<int>(offset + index), boxes[index]);
			}
		}
		if (!infile.good())
		{
			error = "failed while reading hierarchy boxes";
			return false;
		}
	}

	if (requested_depth >= 0)
	{
		if (requested_depth < minimum_candidate_depth || requested_depth > maximum_depth
			|| candidates[requested_depth].empty())
		{
			error = "requested ownership depth is unavailable";
			return false;
		}
		selected_depth = requested_depth;
	}
	else
	{
		selected_depth = minimum_candidate_depth;
		for (int depth = maximum_depth; depth >= minimum_candidate_depth; --depth)
		{
			if (candidates[depth].empty())
				continue;
			std::vector<float> extents;
			extents.reserve(candidates[depth].size());
			for (const auto& candidate : candidates[depth])
				extents.push_back((candidate.second.maxx.head<3>() - candidate.second.minn.head<3>()).maxCoeff());
			const std::size_t middle = extents.size() / 2;
			std::nth_element(extents.begin(), extents.begin() + middle, extents.end());
			selected_depth = depth;
			if (extents[middle] <= target_extent)
				break;
		}
	}

	node_ids.reserve(candidates[selected_depth].size());
	unit_boxes.reserve(candidates[selected_depth].size());
	for (const auto& candidate : candidates[selected_depth])
	{
		node_ids.push_back(candidate.first);
		unit_boxes.push_back(candidate.second);
	}
	if (node_ids.empty())
	{
		error = "hierarchy has no adaptive ownership units";
		return false;
	}
	return true;
}
