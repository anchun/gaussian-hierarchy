#include "spatial_merge.h"
#include "dependencies/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace
{
struct PriorInfo
{
	std::size_t road_points = 0;
	std::size_t skybox_points = 0;
	std::size_t fixed_prior_points = 0;
};

std::vector<std::string> properties(int degree)
{
	std::vector<std::string> result{"x", "y", "z"};
	if (degree == 3)
		result.insert(result.end(), {"nx", "ny", "nz"});
	result.insert(result.end(), {"f_dc_0", "f_dc_1", "f_dc_2"});
	for (int i = 0; i < 3 * ((degree + 1) * (degree + 1) - 1); ++i)
		result.push_back("f_rest_" + std::to_string(i));
	result.insert(result.end(), {"opacity", "scale_0", "scale_1", "scale_2",
		"rot_0", "rot_1", "rot_2", "rot_3"});
	return result;
}

struct Ply
{
	std::size_t count = 0;
	std::size_t stride = 0;
	std::vector<std::size_t> columns;
	std::vector<float> values;

	float at(std::size_t row, std::size_t column) const
	{
		if (columns.at(column) == std::numeric_limits<std::size_t>::max())
			return 0.f;
		return values.at(row * stride + columns.at(column));
	}
};

Ply readPly(const fs::path& path, const std::vector<std::string>& output_properties)
{
	std::ifstream file(path, std::ios::binary);
	if (!file)
		throw std::runtime_error("Cannot open Gaussian PLY: " + path.string());
	std::string line, format, element;
	if (!std::getline(file, line))
		throw std::runtime_error("Invalid PLY signature: " + path.string());
	if (!line.empty() && line.back() == '\r')
		line.pop_back();
	if (line != "ply")
		throw std::runtime_error("Invalid PLY signature: " + path.string());
	Ply result;
	bool vertex_seen = false, end_header = false;
	std::vector<std::string> names;
	while (std::getline(file, line))
	{
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		std::istringstream stream(line);
		std::string keyword;
		stream >> keyword;
		if (keyword == "format")
			std::getline(stream >> std::ws, format);
		else if (keyword == "element")
		{
			std::uint64_t count;
			if (!(stream >> element >> count))
				throw std::runtime_error("Invalid PLY element: " + path.string());
			if (element == "vertex")
			{
				if (vertex_seen || count == 0 || count > fs::file_size(path) / sizeof(float))
					throw std::runtime_error("Invalid PLY vertex count: " + path.string());
				vertex_seen = true;
				result.count = static_cast<std::size_t>(count);
			}
			else if (count != 0)
				throw std::runtime_error("Non-vertex PLY elements are unsupported: " + path.string());
		}
		else if (keyword == "property" && element == "vertex")
		{
			std::string type, name;
			stream >> type >> name;
			if ((type != "float" && type != "float32") || name.empty()
				|| std::find(names.begin(), names.end(), name) != names.end())
				throw std::runtime_error("Expected unique scalar float32 PLY properties: " + path.string());
			names.push_back(name);
		}
		else if (keyword == "end_header")
		{
			end_header = true;
			break;
		}
	}
	const std::uint16_t endian = 1;
	if (!end_header || !vertex_seen || format != "binary_little_endian 1.0"
		|| *reinterpret_cast<const unsigned char*>(&endian) != 1)
		throw std::runtime_error("Expected binary little-endian Gaussian PLY: " + path.string());
	std::size_t rest_count = 0;
	for (const auto& name : names)
		if (name.rfind("f_rest_", 0) == 0)
			++rest_count;
	const auto expected_rest = std::count_if(output_properties.begin(), output_properties.end(),
		[](const std::string& name) { return name.rfind("f_rest_", 0) == 0; });
	if (rest_count != static_cast<std::size_t>(expected_rest))
		throw std::runtime_error("SH properties do not match requested degree: " + path.string());
	for (const auto& name : output_properties)
	{
		const auto found = std::find(names.begin(), names.end(), name);
		if (found == names.end())
		{
			if (name != "nx" && name != "ny" && name != "nz")
				throw std::runtime_error("Missing Gaussian PLY property " + name + ": " + path.string());
			result.columns.push_back(std::numeric_limits<std::size_t>::max());
		}
		else
			result.columns.push_back(std::distance(names.begin(), found));
	}
	result.stride = names.size();
	const auto offset = static_cast<std::uintmax_t>(file.tellg());
	if (result.stride == 0 || result.count > (fs::file_size(path) - offset) / (result.stride * sizeof(float))
		|| offset + result.count * result.stride * sizeof(float) != fs::file_size(path))
		throw std::runtime_error("Unexpected Gaussian PLY data size: " + path.string());
	result.values.resize(result.count * result.stride);
	if (!file.read(reinterpret_cast<char*>(result.values.data()), result.values.size() * sizeof(float)))
		throw std::runtime_error("Truncated Gaussian PLY: " + path.string());
	for (std::size_t row = 0; row < result.count; ++row)
		for (std::size_t column = 0; column < result.columns.size(); ++column)
			if (!std::isfinite(result.at(row, column)))
				throw std::runtime_error("Non-finite Gaussian values: " + path.string());
	return result;
}

json readJson(const fs::path& path)
{
	std::ifstream file(path);
	if (!file)
		throw std::runtime_error("Missing spatial prior metadata: " + path.string());
	json result;
	file >> result;
	return result;
}

PriorInfo priorInfo(const fs::path& directory, std::size_t count)
{
	// A missing explicit prior must not be interpreted as a legacy sky-only prefix.
	const json value = readJson(directory / "prior_info.json");
	for (const char* field : {"road_points", "skybox_points", "fixed_prior_points"})
		if (!value.at(field).is_number_integer() || value.at(field) < 0 || value.at(field) > count)
			throw std::runtime_error("Invalid spatial prior count: " + directory.string());
	PriorInfo result{value.at("road_points").get<std::size_t>(), value.at("skybox_points").get<std::size_t>(),
		value.at("fixed_prior_points").get<std::size_t>()};
	if (result.fixed_prior_points != result.road_points + result.skybox_points)
		throw std::runtime_error("fixed_prior_points does not equal road_points + skybox_points: "
			+ directory.string());
	return result;
}

std::size_t scaffoldCount(const fs::path& directory, const PriorInfo& prior, std::size_t count)
{
	std::ifstream file(directory / "pc_info.txt");
	std::size_t result;
	if (!(file >> result) || result < prior.fixed_prior_points || result > count)
		throw std::runtime_error("Invalid trained scaffold count: " + directory.string());
	return result;
}

int savedIteration(const std::vector<std::string>& directories, int requested)
{
	if (requested > 0)
		return requested;
	int common = -1;
	for (const auto& directory : directories)
	{
		int latest = 0;
		for (const auto& entry : fs::directory_iterator(fs::path(directory) / "point_cloud"))
		{
			const auto name = entry.path().filename().string();
			const auto digits = name.substr(std::min(name.size(), std::size_t(10)));
			if (name.rfind("iteration_", 0) == 0 && !digits.empty()
				&& digits.find_first_not_of("0123456789") == std::string::npos
				&& fs::is_regular_file(entry.path() / "point_cloud.ply"))
				latest = std::max(latest, std::stoi(digits));
		}
		if (latest == 0 || (common > 0 && latest != common))
			throw std::runtime_error("No consistent latest trained PLY iteration; specify --iteration explicitly");
		common = latest;
	}
	return common;
}

void writeJson(const fs::path& path, const json& value)
{
	std::ofstream file(path);
	file.exceptions(std::ios::badbit | std::ios::failbit);
	file << value.dump(2) << '\n';
	file.close();
}
}

void mergeSpatialPlys(const std::vector<std::string>& directories,
	const std::vector<SpatialChunkInfo>& infos, const std::string& output,
	int degree, int requested_iteration)
{
	if (degree < 0 || degree > 3 || directories.empty() || directories.size() != infos.size())
		throw std::runtime_error("Invalid spatial PLY merge arguments");
	const auto fields = properties(degree);
	const auto opacity_column = std::distance(fields.begin(),
		std::find(fields.begin(), fields.end(), "opacity"));
	const auto iteration = savedIteration(directories, requested_iteration);
	const fs::path scaffold = fs::path(directories.front()).parent_path()
		/ "../scaffold/point_cloud/iteration_30000";
	const Ply global = readPly(scaffold / "point_cloud.ply", fields);
	const PriorInfo global_prior = priorInfo(scaffold, global.count);
	std::vector<Ply> chunks;
	std::vector<PriorInfo> priors;
	std::vector<std::vector<std::size_t>> rings(infos.size()), scenes(infos.size());
	std::vector<double> sky(global_prior.skybox_points * fields.size(), 0.);
	double core_frames = 0.;
	for (const auto& info : infos)
		core_frames += info.core_frame_names.size();
	json report_chunks = json::array();
	std::size_t ring_count = 0, scene_count = 0;
	for (std::size_t chunk = 0; chunk < infos.size(); ++chunk)
	{
		const fs::path directory = fs::path(directories[chunk]) / "point_cloud"
			/ ("iteration_" + std::to_string(iteration));
		chunks.push_back(readPly(directory / "point_cloud.ply", fields));
		const Ply& ply = chunks.back();
		priors.push_back(priorInfo(directory, ply.count));
		const PriorInfo& prior = priors.back();
		const auto prefix = scaffoldCount(directory, prior, ply.count);
		if (prior.skybox_points != global_prior.skybox_points)
			throw std::runtime_error("Sky point count differs from global scaffold: " + directory.string());
		const double weight = infos[chunk].core_frame_names.size() / core_frames;
		for (std::size_t row = 0; row < prior.skybox_points; ++row)
			for (std::size_t column = 0; column < fields.size(); ++column)
			{
				const double value = ply.at(prior.road_points + row, column);
				double& target = sky[row * fields.size() + column];
				const auto& field = fields[column];
				const bool geometry = column < 3 || field == "nx" || field == "ny" || field == "nz"
					|| field.rfind("scale_", 0) == 0 || field.rfind("rot_", 0) == 0;
				if (geometry)
				{
					if (chunk != 0 && value != target)
						throw std::runtime_error("Trained sky geometry/order differs: " + fields[column]);
					target = value;
				}
				else if (column == static_cast<std::size_t>(opacity_column))
				{
					const double alpha = value >= 0. ? 1. / (1. + std::exp(-value))
						: std::exp(value) / (1. + std::exp(value));
					target += weight * alpha;
				}
				else
					target += weight * value;
			}
		for (std::size_t row = prior.fixed_prior_points; row < ply.count; ++row)
		{
			const Eigen::Vector2d xy(ply.at(row, 0), ply.at(row, 1));
			if (spatialOwner(infos[chunk], xy) == infos[chunk].chunk_id)
				(row < prefix ? rings[chunk] : scenes[chunk]).push_back(row);
		}
		ring_count += rings[chunk].size();
		scene_count += scenes[chunk].size();
		report_chunks.push_back({
			{"name", fs::path(directories[chunk]).filename().string()},
			{"chunk_id", infos[chunk].chunk_id}, {"core_frames", infos[chunk].core_frame_names.size()},
			{"input_scene_points", ply.count - prefix}, {"input_ring_points", prefix - prior.fixed_prior_points},
			{"retained_scene_points", scenes[chunk].size()}, {"retained_ring_points", rings[chunk].size()}});
		std::cout << "Spatial hard cut: " << report_chunks.back().dump() << std::endl;
	}
	if (scene_count == 0)
		throw std::runtime_error("Spatial merge retained no scene Gaussians");
	for (std::size_t row = 0; row < global_prior.skybox_points; ++row)
	{
		double& alpha = sky[row * fields.size() + opacity_column];
		alpha = std::clamp(alpha, double(std::numeric_limits<float>::epsilon()),
			1. - double(std::numeric_limits<float>::epsilon()));
		alpha = std::log(alpha) - std::log1p(-alpha);
	}
	const std::size_t total = global_prior.fixed_prior_points + ring_count + scene_count;
	const fs::path destination(output);
	if (!destination.parent_path().empty())
		fs::create_directories(destination.parent_path());
	const fs::path temporary(output + ".tmp-" + std::to_string(
		std::chrono::steady_clock::now().time_since_epoch().count()));
	try
	{
		std::ofstream file(temporary, std::ios::binary);
		file.exceptions(std::ios::badbit | std::ios::failbit);
		file << "ply\nformat binary_little_endian 1.0\nelement vertex " << total << '\n';
		for (const auto& field : fields)
			file << "property float " << field << '\n';
		file << "end_header\n";
		std::vector<float> buffer;
		const auto flush = [&]() {
			file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
			buffer.clear();
		};
		const auto append = [&](const Ply& ply, std::size_t row) {
			for (std::size_t column = 0; column < fields.size(); ++column)
				buffer.push_back(ply.at(row, column));
			if (buffer.size() >= 1'000'000)
				flush();
		};
		for (std::size_t row = 0; row < global_prior.road_points; ++row)
			append(global, row);
		flush();
		for (const auto value : sky)
		{
			buffer.push_back(static_cast<float>(value));
			if (buffer.size() >= 1'000'000)
				flush();
		}
		flush();
		for (const auto* groups : {&rings, &scenes})
			for (std::size_t chunk = 0; chunk < chunks.size(); ++chunk)
				for (const auto row : (*groups)[chunk])
					append(chunks[chunk], row);
		flush();
		file.close();
		fs::rename(temporary, destination);
	}
	catch (...)
	{
		fs::remove(temporary);
		throw;
	}
	writeJson(destination.parent_path() / "prior_info.json", {
		{"road_points", global_prior.road_points}, {"skybox_points", global_prior.skybox_points},
		{"fixed_prior_points", global_prior.fixed_prior_points}});
	{
		std::ofstream file(destination.parent_path() / "pc_info.txt");
		file.exceptions(std::ios::badbit | std::ios::failbit);
		file << global_prior.fixed_prior_points + ring_count;
		file.close();
	}
	writeJson(destination.parent_path() / "merge_info.json", {
		{"policy", "spatial_hard_cut"}, {"sky_policy", "core_frame_weighted_sh_and_alpha"},
		{"partition_id", infos.front().partition_id}, {"iteration", iteration}, {"sh_degree", degree},
		{"road_points", global_prior.road_points}, {"skybox_points", global_prior.skybox_points},
		{"ring_points", ring_count}, {"scene_points", scene_count}, {"total_points", total},
		{"chunks", report_chunks}});
	fs::remove(destination.parent_path() / "chunk_info.json");
	std::cout << "Merged spatial PLY: " << total << " float32 Gaussians -> " << output << std::endl;
}
