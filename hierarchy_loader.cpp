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
#include <cstdint>

struct HalfBox2
{
	half_float::half minn[4];
	half_float::half maxx[4];
};

namespace
{
struct HierarchyLayout
{
	int signed_points;
	int points;
	int nodes;
	std::streamoff nodes_offset;
	bool compressed;
};

HierarchyLayout readLayout(std::ifstream& file, const char* filename)
{
	if (!file)
		throw std::runtime_error(std::string("cannot open hierarchy ") + filename);
	file.seekg(0, std::ios_base::end);
	const std::streamoff bytes = file.tellg();
	file.seekg(0);
	HierarchyLayout layout{};
	file.read(reinterpret_cast<char*>(&layout.signed_points), sizeof(int));
	if (!file || layout.signed_points == std::numeric_limits<int>::min())
		throw std::runtime_error(std::string("invalid hierarchy header in ") + filename);
	layout.compressed = layout.signed_points < 0;
	layout.points = std::abs(layout.signed_points);
	if (layout.points == 0)
		throw std::runtime_error(std::string("empty source hierarchy in ") + filename);
	const std::uint64_t point_bytes = layout.compressed
		? sizeof(Eigen::Vector3f) + (4 + 3 + 1 + 48) * sizeof(half_float::half)
		: sizeof(Eigen::Vector3f) * 2 + sizeof(Eigen::Vector4f) + sizeof(float) + sizeof(SHs);
	const std::uint64_t offset = sizeof(int) + point_bytes * static_cast<std::uint64_t>(layout.points);
	if (bytes < 0 || offset + sizeof(int) > static_cast<std::uint64_t>(bytes))
		throw std::runtime_error(std::string("truncated hierarchy Gaussian data in ") + filename);
	file.seekg(static_cast<std::streamoff>(offset));
	file.read(reinterpret_cast<char*>(&layout.nodes), sizeof(int));
	if (!file || layout.nodes <= 0)
		throw std::runtime_error(std::string("empty or invalid hierarchy topology in ") + filename);
	const std::uint64_t node_bytes = layout.compressed
		? sizeof(HalfNode) + sizeof(HalfBox2) : sizeof(Node) + sizeof(Box);
	if (offset + sizeof(int) + node_bytes * static_cast<std::uint64_t>(layout.nodes)
		> static_cast<std::uint64_t>(bytes))
		throw std::runtime_error(std::string("truncated hierarchy topology in ") + filename);
	layout.nodes_offset = static_cast<std::streamoff>(offset + sizeof(int));
	return layout;
}

void validateTopology(const std::vector<Node>& nodes, int points)
{
	if (nodes.empty() || nodes[0].parent != -1)
		throw std::runtime_error("hierarchy has no valid root");
	std::vector<bool> seen(nodes.size(), false);
	std::vector<int> pending{0};
	std::vector<std::pair<int, int>> ranges;
	std::size_t visited = 0;
	std::size_t leaves = 0;
	while (!pending.empty())
	{
		const int index = pending.back();
		pending.pop_back();
		if (seen[index])
			throw std::runtime_error("hierarchy has a cycle or shared child");
		seen[index] = true;
		++visited;
		const Node& node = nodes[index];
		const std::int64_t count = static_cast<std::int64_t>(node.count_leafs) + node.count_merged;
		if (node.depth < 0 || node.count_leafs < 0 || node.count_merged < 0
			|| node.start < 0 || static_cast<std::int64_t>(node.start) + count > points
			|| node.count_children < 0 || node.start_children < 0
			|| static_cast<std::int64_t>(node.start_children) + node.count_children > nodes.size()
			|| (node.depth == 0 && node.count_children != 0)
			|| (node.count_children == 0 && node.count_leafs == 0))
			throw std::runtime_error("hierarchy has an invalid Gaussian or child range");
		if (count > 0)
			ranges.emplace_back(node.start, node.start + static_cast<int>(count));
		leaves += node.count_leafs;
		for (int child = node.start_children; child < node.start_children + node.count_children; ++child)
		{
			if (nodes[child].parent != index || nodes[child].depth >= node.depth)
				throw std::runtime_error("hierarchy child has an invalid parent or depth");
			pending.push_back(child);
		}
	}
	if (visited != nodes.size() || leaves == 0)
		throw std::runtime_error("hierarchy has unreachable nodes or no Gaussian leaves");
	std::sort(ranges.begin(), ranges.end());
	int next = 0;
	for (const auto& range : ranges)
	{
		if (range.first != next)
			throw std::runtime_error("hierarchy Gaussian ranges overlap or leave unreferenced data");
		next = range.second;
	}
	if (next != points)
		throw std::runtime_error("hierarchy contains unreferenced Gaussian data");
}
}

void HierarchyLoader::load(const char* filename,
	std::vector<Eigen::Vector3f>& pos,
	std::vector<SHs>& shs,
	std::vector<float>& alphas,
	std::vector<Eigen::Vector3f>& scales,
	std::vector<Eigen::Vector4f>& rot,
	std::vector<Node>& nodes,
	std::vector<Box>& boxes,
	bool validate_topology)
{
	std::ifstream infile(filename, std::ios_base::binary);
	const HierarchyLayout layout = readLayout(infile, filename);
	infile.seekg(sizeof(int));
	const int P = layout.signed_points;

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

		int N = 0;
		infile.read((char*)&N, sizeof(int));
		if (!infile || N != layout.nodes)
			throw std::runtime_error("failed to read hierarchy node count");

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

		int N = 0;
		infile.read((char*)&N, sizeof(int));
		if (!infile || N != layout.nodes)
			throw std::runtime_error("failed to read hierarchy node count");
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
	if (!infile)
		throw std::runtime_error(std::string("failed to read hierarchy ") + filename);
	if (validate_topology)
		validateTopology(nodes, layout.points);
}

bool HierarchyLoader::loadSpatialFrontier(const char* filename,
	float target_extent,
	std::vector<int>& node_ids,
	std::vector<Box>& unit_boxes,
	std::string& error)
{
	node_ids.clear();
	unit_boxes.clear();
	try
	{
		if (!std::isfinite(target_extent) || target_extent <= 0.f)
			throw std::runtime_error("invalid spatial frontier target extent");
		std::ifstream file(filename, std::ios_base::binary);
		const HierarchyLayout layout = readLayout(file, filename);
		std::vector<Node> nodes(layout.nodes);
		std::vector<Box> boxes(layout.nodes);
		constexpr std::size_t block_size = 1 << 20;
		if (layout.compressed)
		{
			for (std::size_t offset = 0; offset < nodes.size(); offset += block_size)
			{
				const std::size_t count = std::min(block_size, nodes.size() - offset);
				std::vector<HalfNode> block(count);
				file.read(reinterpret_cast<char*>(block.data()), count * sizeof(HalfNode));
				for (std::size_t i = 0; i < count; ++i)
				{
					Node& node = nodes[offset + i];
					node.parent = block[i].parent;
					node.start = block[i].start;
					node.start_children = block[i].start_children;
					node.depth = block[i].dccc[0];
					node.count_children = block[i].dccc[1];
					node.count_leafs = block[i].dccc[2];
					node.count_merged = block[i].dccc[3];
				}
			}
			for (std::size_t offset = 0; offset < boxes.size(); offset += block_size)
			{
				const std::size_t count = std::min(block_size, boxes.size() - offset);
				std::vector<HalfBox2> block(count);
				file.read(reinterpret_cast<char*>(block.data()), count * sizeof(HalfBox2));
				for (std::size_t i = 0; i < count; ++i)
					for (int axis = 0; axis < 4; ++axis)
					{
						boxes[offset + i].minn[axis] = block[i].minn[axis];
						boxes[offset + i].maxx[axis] = block[i].maxx[axis];
					}
			}
		}
		else
		{
			file.read(reinterpret_cast<char*>(nodes.data()), nodes.size() * sizeof(Node));
			file.read(reinterpret_cast<char*>(boxes.data()), boxes.size() * sizeof(Box));
		}
		if (!file)
			throw std::runtime_error("failed to read spatial hierarchy topology");
		validateTopology(nodes, layout.points);
		std::vector<int> pending{0};
		while (!pending.empty())
		{
			const int index = pending.back();
			pending.pop_back();
			const Node& node = nodes[index];
			const Eigen::Vector3f size = boxes[index].maxx.head<3>() - boxes[index].minn.head<3>();
			const bool small = size.allFinite() && (size.array() >= 0.f).all()
				&& size.maxCoeff() <= target_extent;
			// Direct leaves on an internal node cannot be covered by descending into its children.
			if (small || node.count_children == 0 || node.count_leafs > 0)
			{
				node_ids.push_back(index);
				unit_boxes.push_back(boxes[index]);
			}
			else
			{
				for (int i = node.count_children - 1; i >= 0; --i)
					pending.push_back(node.start_children + i);
			}
		}
		return true;
	}
	catch (const std::exception& exception)
	{
		node_ids.clear();
		unit_boxes.clear();
		error = std::string("invalid spatial hierarchy ") + filename + ": " + exception.what();
		return false;
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
