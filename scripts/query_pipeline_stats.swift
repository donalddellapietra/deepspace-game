import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else {
    print("ERROR: No Metal device")
    exit(1)
}
print("device: \(device.name)")
print("registry_id: \(device.registryID)")
print("max_threadgroup_mem: \(device.maxThreadgroupMemoryLength)")
print("max_buffer_len: \(device.maxBufferLength)")
print("max_threads_per_tg: w=\(device.maxThreadsPerThreadgroup.width), h=\(device.maxThreadsPerThreadgroup.height), d=\(device.maxThreadsPerThreadgroup.depth)")
print("")

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("Usage: swift query_pipeline_stats.swift <metallib_path> <entry_point>")
    exit(1)
}
let metallibURL = URL(fileURLWithPath: args[1])
let entryPoint = args[2]

guard let library = try? device.makeLibrary(URL: metallibURL) else {
    print("ERROR: failed to load metallib from \(metallibURL.path)")
    exit(1)
}
print("library_functions: \(library.functionNames)")

guard let function = library.makeFunction(name: entryPoint) else {
    print("ERROR: entry point \(entryPoint) not found")
    exit(1)
}
print("function: \(function.name) type=\(function.functionType.rawValue) (1=Vertex 2=Fragment 3=Kernel)")
print("")

if function.functionType == .kernel {
    do {
        let pipeline = try device.makeComputePipelineState(function: function)
        print("── COMPUTE PIPELINE STATS ──")
        print("max_total_threads_per_threadgroup: \(pipeline.maxTotalThreadsPerThreadgroup)")
        print("thread_execution_width: \(pipeline.threadExecutionWidth)")
        print("static_threadgroup_memory_length: \(pipeline.staticThreadgroupMemoryLength)")
        print("")
        print("── INTERPRETATION ──")
        let maxDeviceThreads = device.maxThreadsPerThreadgroup.width * device.maxThreadsPerThreadgroup.height
        let ratio = Double(pipeline.maxTotalThreadsPerThreadgroup) / Double(maxDeviceThreads)
        print("ratio to device max: \(String(format: "%.2f", ratio * 100))%")
        print("(lower = more registers used per thread; Apple M-series register file supports ~1024 threads at minimal register pressure)")
    } catch {
        print("ERROR: failed to create compute pipeline: \(error)")
        exit(1)
    }
} else {
    print("Note: function is not @kernel (@compute); max_total_threads_per_threadgroup is only meaningful for compute pipelines.")
    print("For fragment shaders, Apple does not expose register count via public Metal API.")
}
