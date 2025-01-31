#ifndef LIBYT_PROJECT_INCLUDE_LOGGING_H_
#define LIBYT_PROJECT_INCLUDE_LOGGING_H_

namespace logging {
void LogInfo(const char* format, ...);
void LogWarning(const char* format, ...);
void LogDebug(const char* format, ...);
void LogError(const char* format, ...);
}  // namespace logging

#endif  // LIBYT_PROJECT_INCLUDE_LOGGING_H_
