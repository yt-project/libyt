# Set Code or User Specific Parameter
## yt\_set\_UserParameter*
```cpp
int yt_set_UserParameterInt   ( const char *key, const int n, const int    *input );
int yt_set_UserParameterLong  ( const char *key, const int n, const long   *input );
int yt_set_UserParameterUint  ( const char *key, const int n, const uint   *input );
int yt_set_UserParameterUlong ( const char *key, const int n, const ulong  *input );
int yt_set_UserParameterFloat ( const char *key, const int n, const float  *input );
int yt_set_UserParameterDouble( const char *key, const int n, const double *input );
int yt_set_UserParameterString( const char *key,              const char   *input );
```
- Usage: Add code or user specific parameters that is used in your input yt [`frontend`](./SetYTParameter.md#yt_param_yt) `XXXDataset` class, since `libyt` will borrow field information (`class XXXFieldInfo`) from it. `libyt` will add them to `libytDataset` class as new attributes.
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
yt_set_UserParameterInt("mhd", 1, &mhd);  
const int srhd = 0;
yt_set_UserParameterInt("srhd", 1, &srhd);
```

