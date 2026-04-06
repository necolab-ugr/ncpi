#include "iaf_bw_2003.h"
#include "nest_extension_interface.h"

namespace mynest
{
class CavallariModule : public nest::NESTExtensionInterface
{
public:
  CavallariModule() {}
  virtual ~CavallariModule() {}

  void initialize() override;
};
}

// Define module instance outside of namespace to avoid name-mangling problems
mynest::CavallariModule cavallari_module_LTX_module;

void
mynest::CavallariModule::initialize()
{
  mynest::register_iaf_bw_2003( "iaf_bw_2003" );
}
