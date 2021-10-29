# Set Code Specific Parameter
## yt\_add\_user\_parameter_*
```cpp
int yt_add_user_parameter_int   ( const char *key, const int n, const int    *input );
int yt_add_user_parameter_long  ( const char *key, const int n, const long   *input );
int yt_add_user_parameter_uint  ( const char *key, const int n, const uint   *input );
int yt_add_user_parameter_ulong ( const char *key, const int n, const ulong  *input );
int yt_add_user_parameter_float ( const char *key, const int n, const float  *input );
int yt_add_user_parameter_double( const char *key, const int n, const double *input );
int yt_add_user_parameter_string( const char *key,              const char   *input );
```
- Usage: Add code specific parameters as new attributes to data set in `yt`. You must add code specific parameters to match your input [`frontend`](./SetYTParameter.md#yt_param_yt), since `libyt` will borrow field information (`class XXXFieldInfo`) from it.
- Return: `YT_SUCCESS` or `YT_FAIL`

## Example
For example in `yt` frontend `gamer`, it uses `mhd`, `srhd`, etc. Code snippet of `gamer` frontend:
```python
# yt/frontends/gamer/io.py
class GAMERFieldInfo(FieldInfoContainer):
    def setup_fluid_fields(self):
        ...
        if self.ds.srhd:
            ...
```

We must set these parameters to match the field information in this frontend.
```cpp
/* Since we set frontend to "gamer", we should set the code specific parameter
   mhd and srhd here. */
const int mhd = 0; 
yt_add_user_parameter_int("mhd", 1, &mhd);  
const int srhd = 0;  
yt_add_user_parameter_int("srhd", 1, &srhd);
```

