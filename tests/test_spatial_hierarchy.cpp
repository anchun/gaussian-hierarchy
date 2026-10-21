#include "writer.h"
#include "loader.h"
#include "spatial_merge.h"
#include "dependencies/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace
{
void require(bool condition, const std::string& message)
{
	if (!condition)
		throw std::runtime_error(message);
}

struct TestFiles
{
	fs::path directory = "test_spatial_hierarchy_data";
	TestFiles()
	{
		require(!fs::exists(directory), "test directory already exists");
		fs::create_directory(directory);
	}
	~TestFiles() { fs::remove_all(directory); }
};

Gaussian gaussian(float x)
{
	Gaussian result;
	result.position = Eigen::Vector3f(x, 0.f, 10.f);
	result.scale = Eigen::Vector3f::Constant(0.05f);
	result.rotation = Eigen::Vector4f(1.f, 0.f, 0.f, 0.f);
	result.opacity = 0.75f;
	result.shs.setZero();
	computeCovariance(result.scale, result.rotation, result.covariance);
	return result;
}

void writeJson(const fs::path& path, const json& value)
{
	fs::create_directories(path.parent_path());
	std::ofstream(path) << value.dump(2);
}

std::string quoted(const fs::path& path)
{
	return "\"" + path.string() + "\"";
}
}

// End-to-end check of the merger entry point: spatial chunks are merged by hard BSP ownership.
int main()
{
	TestFiles files;
	std::string error;
	std::ifstream fixture_file(SPATIAL_FIXTURE_PATH);
	json fixture;
	fixture_file >> fixture;
	const fs::path sources = files.directory / "sources";
	const fs::path trained = files.directory / "trained" / "chunks";
	const fs::path scaffold = trained.parent_path() / "scaffold" / "point_cloud" / "iteration_30000";
	fs::create_directories(scaffold);
	Writer::writePly((scaffold / "point_cloud.ply").string().c_str(), {gaussian(100)}, 3);
	std::ofstream(scaffold / "pc_info.txt") << "0\n";
	writeJson(scaffold / "prior_info.json", {{"road_points", 0}, {"skybox_points", 0}, {"fixed_prior_points", 0}});
	for (int chunk = 0; chunk < 3; ++chunk)
	{
		const std::string name = std::to_string(chunk) + "_0";
		json document = fixture;
		document["chunk_id"] = chunk;
		document["core_frame_names"] = {std::to_string(chunk) + ".jpg"};
		writeJson(sources / name / "chunk_info.json", document);
		writeJson(trained / name / "chunk_info.json", document);
	}
	const std::string chunks = " 2_0 1_0 0_0";
	for (const bool duplicates : {true, false})
	{
		std::vector<float> expected;
		for (int chunk = 0; chunk < 3; ++chunk)
		{
			const std::string name = std::to_string(chunk) + "_0";
			std::vector<Gaussian> gaussians{gaussian(duplicates ? -5.f : -5.f - 5.f * chunk)};
			const fs::path cloud = trained / name / "point_cloud" / "iteration_42000";
			fs::create_directories(cloud);
			Writer::writePly((cloud / "point_cloud.ply").string().c_str(), gaussians, 3);
			std::ofstream(cloud / "pc_info.txt") << "0";
			writeJson(cloud / "prior_info.json",
				{{"road_points", 0}, {"skybox_points", 0}, {"fixed_prior_points", 0}});
			SpatialChunkInfo info;
			require(loadSpatialChunkInfo((sources / name / "chunk_info.json").string(), info, error), error);
			if (spatialOwner(info, gaussians[0].position.head<2>().cast<double>()) == chunk)
				expected.push_back(gaussians[0].position.x());
		}
		std::sort(expected.begin(), expected.end());
		const fs::path output = files.directory / "merged.ply";
		const std::string command = quoted(SPATIAL_MERGER_PATH) + " " + quoted(trained) + " 3 "
			+ quoted(sources) + " " + quoted(output) + chunks;
		require(std::system(command.c_str()) == 0, "spatial merger entry point failed");
		std::vector<Gaussian> merged;
		Loader::loadPly(output.string().c_str(), merged);
		std::vector<float> actual;
		for (const Gaussian& g : merged)
		{
			actual.push_back(g.position.x());
			require(std::abs(g.opacity - 0.75f) < 1e-6f, "PLY spatial opacity was attenuated");
		}
		std::sort(actual.begin(), actual.end());
		require(actual == expected, "merger output disagrees with its spatial ownership policy");
	}
	const std::string hierarchy_command = quoted(SPATIAL_MERGER_PATH) + " " + quoted(trained) + " 3 "
		+ quoted(sources) + " " + quoted(files.directory / "merged.hier") + chunks;
	require(std::system(hierarchy_command.c_str()) != 0, "spatial hierarchy output was not rejected");
}
