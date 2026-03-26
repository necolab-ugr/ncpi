/*
 *  iaf_bw_2003.cpp
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
 * Author:
 * Pablo Martinez-Canada (pablo.martinez@iit.it)
 */

#include "iaf_bw_2003.h"

#ifdef HAVE_GSL

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

#include "dict_util.h"
#include "exceptions.h"
#include "kernel_manager.h"
#include "nest_impl.h"
#include "numerics.h"
#include "universal_data_logger_impl.h"

using namespace nest;

namespace
{
template < typename ValueType >
void
update_value_param_compat( const mynest::StatusDictionary& d, const Name& name, ValueType& value, nest::Node* node )
{
#ifdef CAVALLARI_USE_LEGACY_NEST_API
  updateValueParam< ValueType >( d, name, value, node );
#else
  update_value_param( d, name, value, node );
#endif
}

template < typename ValueType >
void
dict_set_compat( mynest::StatusDictionary& d, const Name& name, const ValueType& value )
{
#ifdef CAVALLARI_USE_LEGACY_NEST_API
  ( *d )[ name ] = value;
#else
  d[ name ] = value;
#endif
}
}

nest::RecordablesMap< mynest::iaf_bw_2003 > mynest::iaf_bw_2003::recordablesMap_;

void
mynest::register_iaf_bw_2003( const std::string& name )
{
  nest::register_node_model< iaf_bw_2003 >( name );
}

namespace nest
{
template <>
void
RecordablesMap< mynest::iaf_bw_2003 >::create()
{
  insert_( names::V_m, &mynest::iaf_bw_2003::get_y_elem_< mynest::iaf_bw_2003::State_::V_M > );
  insert_( names::g_ex, &mynest::iaf_bw_2003::get_y_elem_< mynest::iaf_bw_2003::State_::G_EXC > );
  insert_( names::g_in, &mynest::iaf_bw_2003::get_y_elem_< mynest::iaf_bw_2003::State_::G_INH > );
  insert_( names::t_ref_remaining, &mynest::iaf_bw_2003::get_r_ );
}
}

extern "C" inline int
mynest::iaf_bw_2003_dynamics( double, const double y[], double f[], void* pnode )
{
  typedef mynest::iaf_bw_2003::State_ S;

  assert( pnode );
  const mynest::iaf_bw_2003& node = *( reinterpret_cast< mynest::iaf_bw_2003* >( pnode ) );

  const double& V = y[ S::V_M ];

  const double I_syn_exc = y[ S::G_EXC ] * ( V - node.P_.E_ex );
  const double I_syn_inh = y[ S::G_INH ] * ( V - node.P_.E_in );
  const double I_leak = node.P_.g_L * ( V - node.P_.E_L );

  f[ 0 ] = ( -I_leak - I_syn_exc - I_syn_inh + node.B_.I_stim_ + node.P_.I_e ) / node.P_.C_m;
  f[ 1 ] = -y[ S::DG_EXC ] / node.P_.tau_decay_AMPA;
  f[ 2 ] = y[ S::DG_EXC ] - ( y[ S::G_EXC ] / node.P_.tau_rise_AMPA );
  f[ 3 ] = -y[ S::DG_INH ] / node.P_.tau_decay_GABA_A;
  f[ 4 ] = y[ S::DG_INH ] - ( y[ S::G_INH ] / node.P_.tau_rise_GABA_A );

  return GSL_SUCCESS;
}

mynest::iaf_bw_2003::Parameters_::Parameters_()
  : V_th( -52.0 )
  , V_reset( -59.0 )
  , t_ref( 2.0 )
  , g_L( 25.0 )
  , C_m( 500.0 )
  , E_ex( 0.0 )
  , E_in( -80.0 )
  , E_L( -70.0 )
  , tau_rise_AMPA( 0.4 )
  , tau_decay_AMPA( 2.0 )
  , tau_rise_GABA_A( 0.25 )
  , tau_decay_GABA_A( 5.0 )
  , tau_m( 20.0 )
  , I_e( 0.0 )
{
}

mynest::iaf_bw_2003::State_::State_( const Parameters_& p )
  : r( 0 )
{
  y[ V_M ] = p.E_L;
  for ( size_t i = 1; i < STATE_VEC_SIZE; ++i )
  {
    y[ i ] = 0;
  }
}

mynest::iaf_bw_2003::State_::State_( const State_& s )
  : r( s.r )
{
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y[ i ] = s.y[ i ];
  }
}

mynest::iaf_bw_2003::State_&
mynest::iaf_bw_2003::State_::operator=( const State_& s )
{
  if ( this == &s )
  {
    return *this;
  }
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y[ i ] = s.y[ i ];
  }

  r = s.r;
  return *this;
}

mynest::iaf_bw_2003::Buffers_::Buffers_( mynest::iaf_bw_2003& n )
  : logger_( n )
  , s_( nullptr )
  , c_( nullptr )
  , e_( nullptr )
{
}

mynest::iaf_bw_2003::Buffers_::Buffers_( const Buffers_&, mynest::iaf_bw_2003& n )
  : logger_( n )
  , s_( nullptr )
  , c_( nullptr )
  , e_( nullptr )
{
}

void
mynest::iaf_bw_2003::Parameters_::get( StatusDictionary& d ) const
{
  dict_set_compat( d, names::V_th, V_th );
  dict_set_compat( d, names::V_reset, V_reset );
  dict_set_compat( d, names::t_ref, t_ref );
  dict_set_compat( d, names::g_L, g_L );
  dict_set_compat( d, names::E_L, E_L );
  dict_set_compat( d, names::E_ex, E_ex );
  dict_set_compat( d, names::E_in, E_in );
  dict_set_compat( d, names::C_m, C_m );
  dict_set_compat( d, names::tau_rise_AMPA, tau_rise_AMPA );
  dict_set_compat( d, names::tau_decay_AMPA, tau_decay_AMPA );
  dict_set_compat( d, names::tau_rise_GABA_A, tau_rise_GABA_A );
  dict_set_compat( d, names::tau_decay_GABA_A, tau_decay_GABA_A );
  dict_set_compat( d, names::tau_m, tau_m );
  dict_set_compat( d, names::I_e, I_e );
}

void
mynest::iaf_bw_2003::Parameters_::set( const StatusDictionary& d, Node* node )
{
  update_value_param_compat( d, names::V_th, V_th, node );
  update_value_param_compat( d, names::V_reset, V_reset, node );
  update_value_param_compat( d, names::t_ref, t_ref, node );
  update_value_param_compat( d, names::E_L, E_L, node );
  update_value_param_compat( d, names::E_ex, E_ex, node );
  update_value_param_compat( d, names::E_in, E_in, node );
  update_value_param_compat( d, names::C_m, C_m, node );
  update_value_param_compat( d, names::g_L, g_L, node );
  update_value_param_compat( d, names::tau_rise_AMPA, tau_rise_AMPA, node );
  update_value_param_compat( d, names::tau_decay_AMPA, tau_decay_AMPA, node );
  update_value_param_compat( d, names::tau_rise_GABA_A, tau_rise_GABA_A, node );
  update_value_param_compat( d, names::tau_decay_GABA_A, tau_decay_GABA_A, node );
  update_value_param_compat( d, names::tau_m, tau_m, node );
  update_value_param_compat( d, names::I_e, I_e, node );

  if ( V_reset >= V_th )
  {
    throw BadProperty( "Reset potential must be smaller than threshold." );
  }
  if ( C_m <= 0 )
  {
    throw BadProperty( "Capacitance must be strictly positive." );
  }
  if ( t_ref < 0 )
  {
    throw BadProperty( "Refractory time cannot be negative." );
  }
  if ( tau_rise_AMPA <= 0 || tau_decay_AMPA <= 0 || tau_rise_GABA_A <= 0 || tau_decay_GABA_A <= 0
    || tau_m <= 0 )
  {
    throw BadProperty( "All time constants must be strictly positive." );
  }
}

void
mynest::iaf_bw_2003::State_::get( StatusDictionary& d ) const
{
  dict_set_compat( d, names::V_m, y[ V_M ] );
}

void
mynest::iaf_bw_2003::State_::set( const StatusDictionary& d, const Parameters_&, Node* node )
{
  update_value_param_compat( d, names::V_m, y[ V_M ], node );
}

mynest::iaf_bw_2003::iaf_bw_2003()
  : ArchivingNode()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

mynest::iaf_bw_2003::iaf_bw_2003( const mynest::iaf_bw_2003& n )
  : ArchivingNode( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

mynest::iaf_bw_2003::~iaf_bw_2003()
{
  if ( B_.s_ )
  {
    gsl_odeiv_step_free( B_.s_ );
  }
  if ( B_.c_ )
  {
    gsl_odeiv_control_free( B_.c_ );
  }
  if ( B_.e_ )
  {
    gsl_odeiv_evolve_free( B_.e_ );
  }
}

void
mynest::iaf_bw_2003::init_buffers_()
{
  ArchivingNode::clear_history();

  B_.spike_exc_.clear();
  B_.spike_inh_.clear();
  B_.currents_.clear();

  B_.logger_.reset();

  B_.step_ = Time::get_resolution().get_ms();
  B_.IntegrationStep_ = B_.step_;

  if ( B_.s_ == nullptr )
  {
    B_.s_ = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_step_reset( B_.s_ );
  }

  if ( B_.c_ == nullptr )
  {
    B_.c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  }
  else
  {
    gsl_odeiv_control_init( B_.c_, 1e-3, 0.0, 1.0, 0.0 );
  }

  if ( B_.e_ == nullptr )
  {
    B_.e_ = gsl_odeiv_evolve_alloc( State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_evolve_reset( B_.e_ );
  }

  B_.sys_.function = iaf_bw_2003_dynamics;
  B_.sys_.jacobian = nullptr;
  B_.sys_.dimension = State_::STATE_VEC_SIZE;
  B_.sys_.params = reinterpret_cast< void* >( this );

  B_.I_stim_ = 0.0;
}

double
mynest::iaf_bw_2003::get_normalisation_factor( double tau_rise, double tau_decay, double tau_m )
{
  const double t_peak = ( tau_decay * tau_rise ) * std::log( tau_decay / tau_rise ) / ( tau_decay - tau_rise );
  const double prefactor = ( 1 / tau_rise ) - ( 1 / tau_decay );
  const double peak_value = ( std::exp( -t_peak / tau_decay ) - std::exp( -t_peak / tau_rise ) );
  const double g_peak = ( tau_m / tau_decay ) * pow( tau_rise / tau_decay, tau_rise / ( tau_decay - tau_rise ) );

  return g_peak * prefactor / peak_value;
}

void
mynest::iaf_bw_2003::pre_run_hook()
{
  B_.logger_.init();

  V_.PSConInit_E = mynest::iaf_bw_2003::get_normalisation_factor( P_.tau_rise_AMPA, P_.tau_decay_AMPA, P_.tau_m );
  V_.PSConInit_I =
    mynest::iaf_bw_2003::get_normalisation_factor( P_.tau_rise_GABA_A, P_.tau_decay_GABA_A, P_.tau_m );
  V_.RefractoryCounts = Time( Time::ms( P_.t_ref ) ).get_steps();

  assert( V_.RefractoryCounts >= 0 );
}

void
mynest::iaf_bw_2003::update( Time const& origin, const long from, const long to )
{
  assert( to >= 0 && from < static_cast< long >( kernel().connection_manager.get_min_delay() ) );
  assert( from < to );

  for ( long lag = from; lag < to; ++lag )
  {
    double t = 0.0;

    while ( t < B_.step_ )
    {
      const int status = gsl_odeiv_evolve_apply(
        B_.e_, B_.c_, B_.s_, &B_.sys_, &t, B_.step_, &B_.IntegrationStep_, S_.y );
      if ( status != GSL_SUCCESS )
      {
        throw GSLSolverFailure( get_name(), status );
      }
    }

    if ( S_.r )
    {
      --S_.r;
      S_.y[ State_::V_M ] = P_.V_reset;
    }
    else if ( S_.y[ State_::V_M ] >= P_.V_th )
    {
      S_.r = V_.RefractoryCounts;
      S_.y[ State_::V_M ] = P_.V_reset;

      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );
    }

    S_.y[ State_::DG_EXC ] += B_.spike_exc_.get_value( lag ) * V_.PSConInit_E;
    S_.y[ State_::DG_INH ] += B_.spike_inh_.get_value( lag ) * V_.PSConInit_I;
    B_.I_stim_ = B_.currents_.get_value( lag );
    B_.logger_.record_data( origin.get_steps() + lag );
  }
}

void
mynest::iaf_bw_2003::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  if ( e.get_weight() > 0.0 )
  {
    B_.spike_exc_.add_value(
      e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
      e.get_weight() * e.get_multiplicity() );
  }
  else
  {
    B_.spike_inh_.add_value(
      e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
      -e.get_weight() * e.get_multiplicity() );
  }
}

void
mynest::iaf_bw_2003::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_current() );
}

void
mynest::iaf_bw_2003::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

void
mynest::iaf_bw_2003::get_status( StatusDictionary& d ) const
{
  P_.get( d );
  S_.get( d );
  nest::ArchivingNode::get_status( d );
  dict_set_compat( d, nest::names::recordables, recordablesMap_.get_list() );
}

void
mynest::iaf_bw_2003::set_status( const StatusDictionary& d )
{
  Parameters_ ptmp = P_;
  ptmp.set( d, this );
  State_ stmp = S_;
  stmp.set( d, ptmp, this );
  nest::ArchivingNode::set_status( d );
  P_ = ptmp;
  S_ = stmp;
}

#endif
