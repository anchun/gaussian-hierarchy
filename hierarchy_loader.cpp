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
