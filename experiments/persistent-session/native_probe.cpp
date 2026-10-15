#include <cstddef>
#include <new>
extern "C" float dojo_dependency_add(float, float);
#ifndef PROBE_VERSION
#define PROBE_VERSION 1
#endif
struct State { unsigned long long calls; };
extern "C" {
void* fixture_new() { return new (std::nothrow) State{0}; }
void fixture_delete(void* handle) { delete static_cast<State*>(handle); }
unsigned long long fixture_calls(void* handle) { return static_cast<State*>(handle)->calls; }
int fixture_version() { return PROBE_VERSION; }
void fixture_apply(void* handle, float* values, std::size_t size) {
    ++static_cast<State*>(handle)->calls;
    for (std::size_t i = 0; i < size; ++i)
        values[i] = dojo_dependency_add(values[i], PROBE_VERSION);
}
}
