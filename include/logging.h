#ifndef LIBYT_PROJECT_INCLUDE_LOGGING_H_
#define LIBYT_PROJECT_INCLUDE_LOGGING_H_

namespace logging {
void LogInfo(const char* format, ...);
void LogWarning(const char* format, ...);
void LogDebug(const char* format, ...);
void LogError(const char* format, ...);
}  // namespace logging

#define YT_ABORT(...)                                                                    \
  {                                                                                      \
    logging::LogError(__VA_ARGS__);                                                      \
    fprintf(stderr,                                                                      \
            "%13s==> file <%s>, line <%d>, function <%s>\n",                             \
            "",                                                                          \
            __FILE__,                                                                    \
            __LINE__,                                                                    \
            __FUNCTION__);                                                               \
    return YT_FAIL;                                                                      \
  }

#endif  // LIBYT_PROJECT_INCLUDE_LOGGING_H_
