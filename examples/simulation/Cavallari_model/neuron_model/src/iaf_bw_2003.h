/*
 *  iaf_bw_2003.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef IAF_BW_2003_H
#define IAF_BW_2003_H

#include "config.h"

#ifdef HAVE_GSL

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

#include "archiving_node.h"
#include "connection.h"
#ifdef CAVALLARI_USE_LEGACY_NEST_API
#include "dictdatum.h"
#else
#include "dict.h"
#endif
#include "event.h"
#include "nest_types.h"
#include "recordables_map.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

namespace mynest
{
void register_iaf_bw_2003( const std::string& name );

extern "C" int iaf_bw_2003_dynamics( double, const double*, double*, void* );

#ifdef CAVALLARI_USE_LEGACY_NEST_API
using StatusDictionary = DictionaryDatum;
#else
using StatusDictionary = Dictionary;
#endif

/** @BeginDocumentation
Name: iaf_bw_2003 - Integrate-and-fire neuron model with conductance-based
                    synapse described by a delayed difference of exponentials [1,2].

References:
[1] Brunel, N., & Wang, X. J. (2003). What determines the frequency of fast network
oscillations with irregular neural discharges? I. Synaptic dynamics and
excitation-inhibition balance. Journal of neurophysiology, 90(1), 415-430.

[2] Cavallari, S., Panzeri, S., & Mazzoni, A. (2014). Comparison of the dynamics
of neural interactions between current-based and conductance-based integrate-and-fire
recurrent networks. Frontiers in neural circuits, 8, 12.

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

Author:
Pablo Martinez-Canada (pablo.martinez@iit.it), based on iaf_cond_beta

SeeAlso: iaf_cond_beta
*/

class iaf_bw_2003 : public nest::ArchivingNode
{
public:
  iaf_bw_2003();
  iaf_bw_2003( const iaf_bw_2003& );
  ~iaf_bw_2003() override;

  using nest::Node::handle;
  using nest::Node::handles_test_event;

  size_t send_test_event( nest::Node&, size_t, nest::synindex, bool ) override;

  void handle( nest::SpikeEvent& ) override;
  void handle( nest::CurrentEvent& ) override;
  void handle( nest::DataLoggingRequest& ) override;

  size_t handles_test_event( nest::SpikeEvent&, size_t ) override;
  size_t handles_test_event( nest::CurrentEvent&, size_t ) override;
  size_t handles_test_event( nest::DataLoggingRequest&, size_t ) override;

  void get_status( StatusDictionary& ) const override;
  void set_status( const StatusDictionary& ) override;

private:
  void init_buffers_() override;
  double get_normalisation_factor( double, double, double );
  void pre_run_hook() override;
  void update( nest::Time const&, const long, const long ) override;

  friend int iaf_bw_2003_dynamics( double, const double*, double*, void* );
  friend class nest::RecordablesMap< iaf_bw_2003 >;
  friend class nest::UniversalDataLogger< iaf_bw_2003 >;

  struct Parameters_
  {
    double V_th;
    double V_reset;
    double t_ref;
    double g_L;
    double C_m;
    double E_ex;
    double E_in;
    double E_L;
    double tau_rise_AMPA;
    double tau_decay_AMPA;
    double tau_rise_GABA_A;
    double tau_decay_GABA_A;
    double tau_m;
    double I_e;

    Parameters_();
    void get( StatusDictionary& ) const;
    void set( const StatusDictionary&, nest::Node* );
  };

  struct State_
  {
    enum StateVecElems
    {
      V_M = 0,
      DG_EXC,
      G_EXC,
      DG_INH,
      G_INH,
      STATE_VEC_SIZE
    };

    double y[ STATE_VEC_SIZE ];
    int r;

    State_( const Parameters_& );
    State_( const State_& );
    State_& operator=( const State_& );

    void get( StatusDictionary& ) const;
    void set( const StatusDictionary&, const Parameters_&, nest::Node* );
  };

  struct Buffers_
  {
    Buffers_( iaf_bw_2003& );
    Buffers_( const Buffers_&, iaf_bw_2003& );

    nest::RingBuffer spike_exc_;
    nest::RingBuffer spike_inh_;
    nest::RingBuffer currents_;
    nest::UniversalDataLogger< iaf_bw_2003 > logger_;
    gsl_odeiv_step* s_;
    gsl_odeiv_control* c_;
    gsl_odeiv_evolve* e_;
    gsl_odeiv_system sys_;
    double step_;
    double IntegrationStep_;
    double I_stim_;
  };

  struct Variables_
  {
    double PSConInit_E;
    double PSConInit_I;
    int RefractoryCounts;
  };

  template < State_::StateVecElems elem >
  double
  get_y_elem_() const
  {
    return S_.y[ elem ];
  }

  double
  get_r_() const
  {
    return nest::Time::get_resolution().get_ms() * S_.r;
  }

  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;

  static nest::RecordablesMap< iaf_bw_2003 > recordablesMap_;
};

inline size_t
mynest::iaf_bw_2003::send_test_event( nest::Node& target, size_t receptor_type, nest::synindex, bool )
{
  nest::SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline size_t
mynest::iaf_bw_2003::handles_test_event( nest::SpikeEvent&, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline size_t
mynest::iaf_bw_2003::handles_test_event( nest::CurrentEvent&, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline size_t
mynest::iaf_bw_2003::handles_test_event( nest::DataLoggingRequest& dlr, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

} // namespace mynest

#endif

#endif
