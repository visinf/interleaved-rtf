#!/usr/bin/perl
use Data::Dumper;

my $Name = shift @ARGV;

$/ = undef;
my $text = <>;
chop $text;

if( $text =~ !/^[+-]/ ) {
    $text = "+$text";
}
$text =~ s/([+-])([a-z])/${1}1 ${2}/g;

print STDOUT "// Automatically generated for polynomial:\n// $text\n";

my @terms = split /(?=[+-])/, $text;

#print STDERR Dumper(\@terms) . "\n";
#exit 0;

for( my $i = 0; $i < @terms; ++$i ) {
    my $factors = $terms[$i];
    my @factors = split /\s+/, $factors;

    my @left = grep { !/\_j/ } @factors;
    my @right = grep { /_j/ } @factors;

    push( @left, '1' ) if( @left == 0 );
    push( @right, '1' ) if( @right == 0 );

    my @rights = sort @right;
    my @lefts = sort @left;

    $terms[$i] = { 'left' => \@lefts, 'right' => \@rights };
}

#print STDOUT Dumper(@terms) . "\n";

my %right = ();

for( my $i = 0; $i < @terms; ++$i ) {
     my $rright = $terms[$i]->{'right'};
     my $rleft  = $terms[$i]->{'left'};

     my @rights = @{$rright};

     my $righttext = join ( ' ', @rights  );

     push( @{$right{$righttext}}, $rleft );
}

BoilerPlate();
AddInLinearContribution();
print STDOUT "\n";
AddInImplicitMatrixMultipliedBy();
print STDOUT "\n";
AccumulateGradient();
print STDOUT "};\n";

sub order
{
    my $str = shift;
    return "" if $str eq "1" or $str =~ m/^[+-]/;

    if( $str =~ m/\^([0-9]+)/ ) {
#	print STDERR "Order of $str = $1\n";
	return $1;
    } else {
	return 1;
    }
}

sub base
{
    my $str = shift;

    if( $str =~ m/^([a-z])/ ) {
	return $1;
    } else {
	return "one" if( $str eq "1" );
	return $str if ( $str =~ m/^[+-]/ );
	die "Couldn't find base of expression $str";
    }
}

sub baseval
{
    my $str = shift;

    if( $str =~ m/^([a-z])/ ) {
	return $1;
    } else {
	return $str;
    }
}

sub pix
{
    my $str = shift;

    if( $str =~ m/_([a-z0-9])/ ) {
	return $1;
    } else {
	return "";
    }
}


sub InitializeFactors
{
    my %terms = ();

    for( my $i = 0; $i < @terms; ++$i ) {
	my $rright = $terms[$i]->{'right'};
	my $rleft  = $terms[$i]->{'left'};

	my @all = ();
	push(@all, @$rright);
	push(@all, @$rleft);

	for my $term( @all ) {
	    $terms{$term}++ if $term =~ m/_/;
	}
    } 
    my @sorted = sort { order($a) <=> order($b)  } keys %terms;
#    print STDERR Dumper(\@sorted) . "\n";

    for my $term( @sorted ) {
	my $base = base($term);
	my $order = order($term);
	my $pix = pix($term);

	if( $order == 1 ) {
	    print STDOUT "const auto $base$order$pix = TFeature::QuadraticBasis(x, $pix, $base);\n";
	} else {
	    my $op = join(" * ", (("${base}1${pix}") x $order));
	    print STDOUT "const auto $base$order$pix = $op;\n";
	}
    }
    print STDOUT "\n";
}

sub Declarations
{
    my $incy = shift;

    for my $r( keys %right ) {
	my @ts = split /\s+/, $r;

	my $id = "";
	for my $term ( @ts ) {
#	    print STDOUT "$term\n";
	    my $base = base($term);
	    my $order = order($term);
	    my $pix = pix($term);
	    
	    $id .= "$base$order$pix";
	}

	if( defined($incy) && $incy ) {
	    print STDOUT "TVarVector ${id}Ty = TVarVector::Zero();\n";
	}
	print STDOUT "TValue ${id}Tone = TValue(0);\n";
    }
    print STDOUT "\n";
}

sub maketerm
{
    my $rfactors = shift;

    my $term = "(";
    my @normfactors = map { base($_) . order($_) . pix($_) } @$rfactors;
    $term .= join(" * ", @normfactors);
    $term .= ")";
    return $term;
}

sub QuadraticUpdates
{
    my ($cov, $var) = @_;

    my $covexpr = "Fy[i] += $cov * (";
    my $varexpr = "Fy[i] += $var * ( (";

    for my $r( keys %right ) {
	my @ts = split /\s+/, $r;

	my $id = "";
	my $val = "";
	for my $term ( @ts ) {
#	    print STDOUT "$term\n";
	    my $base = base($term);
	    my $order = order($term);
	    my $pix = pix($term);
	    
	    $id .= "$base$order$pix";
	    $val .= baseval($term) . "$order$pix * ";
	}
	chop $val; chop $val; chop $val;

	print STDOUT "${id}Ty += (scale * $val) * y[j];\n";
	print STDOUT "${id}Tone += scale * $val;\n";

	my $rleft = $right{$r};
	my $expr = "( ";
	my @terms = map { maketerm($_) } @$rleft;
	$expr .= join( ' + ' , @terms);
	$expr .= " )";
	$covexpr .= "$expr * ${id}Ty + ";
	$varexpr .= "$expr * ${id}Tone + ";
    }
    chop $covexpr; chop $covexpr; chop $covexpr;
    chop $varexpr; chop $varexpr; chop $varexpr;
    $covexpr .= ")";
    $varexpr .= ") * y[i])";
    print STDOUT "\n";
    print STDOUT "${covexpr};\n";
    print STDOUT "${varexpr};\n";
}

sub QuadraticGradientUpdates
{
    my ($cov, $var) = @_;

    my $covexpr = "$cov += c[i] * (";
    my $varexpr = "$var += c[i] * ( (";

    for my $r( keys %right ) {
	my @ts = split /\s+/, $r;

	my $id = "";
	my $val = "";
	for my $term ( @ts ) {
	    my $base = base($term);
	    my $order = order($term);
	    my $pix = pix($term);
	    
	    $id .= "$base$order$pix";
	    $val .= baseval($term) . "$order$pix * ";
	}
	chop $val; chop $val; chop $val;

	print STDOUT "${id}Ty += (scale * $val) * y[j];\n";
	print STDOUT "${id}Tone += scale * $val;\n";

	my $rleft = $right{$r};
	my $expr = "( ";
	my @terms = map { maketerm($_) } @$rleft;
	$expr .= join( ' + ' , @terms);
	$expr .= " )";
	$covexpr .= "$expr * ${id}Ty + ";
	$varexpr .= "$expr * ${id}Tone + ";
    }
    chop $covexpr; chop $covexpr; chop $covexpr;
    chop $varexpr; chop $varexpr; chop $varexpr;
    $covexpr .= ").transpose()";
    $varexpr .= ") * y[i]).transpose()";
    print STDOUT "\n";
    print STDOUT "${covexpr};\n";
    print STDOUT "${varexpr};\n";
}

sub LinearUpdates
{
    my ($seg) = @_;

    my $segexpr = "l[i] += (";

    for my $r( keys %right ) {
	my @ts = split /\s+/, $r;

	my $id = "";
	my $val = "";
	for my $term ( @ts ) {
	    my $base = base($term);
	    my $order = order($term);
	    my $pix = pix($term);
	    
	    $id .= "$base$order$pix";
	    $val .= baseval($term) . "$order$pix * ";
	}
	chop $val; chop $val; chop $val;

	print STDOUT "${id}Tone += scale * $val;\n";

	my $rleft = $right{$r};
	my $expr = "( ";
	my @terms = map { maketerm($_) } @$rleft;
	$expr .= join( ' + ' , @terms);
	$expr .= " )";
	$segexpr .= "$expr * ${id}Tone + ";
    }
    chop $segexpr; chop $segexpr; chop $segexpr;
    $segexpr .= ") * $seg";
    print STDOUT "\n";
    print STDOUT "${segexpr};\n";
}

sub LinearGradientUpdates
{
    my ($seg) = @_;

    my $segexpr = "$seg += c[i] * (";

    for my $r( keys %right ) {
	my @ts = split /\s+/, $r;

	my $id = "";
	my $val = "";
	for my $term ( @ts ) {
	    my $base = base($term);
	    my $order = order($term);
	    my $pix = pix($term);
	    
	    $id .= "$base$order$pix";
	    $val .= baseval($term) . "$order$pix * ";
	}
	chop $val; chop $val; chop $val;

	print STDOUT "${id}Tone += scale * $val;\n";

	my $rleft = $right{$r};
	my $expr = "( ";
	my @terms = map { maketerm($_) } @$rleft;
	$expr .= join( ' + ' , @terms);
	$expr .= " )";
	$segexpr .= "$expr * ${id}Tone + ";
    }
    chop $segexpr; chop $segexpr; chop $segexpr;
    $segexpr .= ")";
    print STDOUT "\n";
    print STDOUT "${segexpr};\n";
}

sub AddInLinearContribution
{
    print STDOUT <<'HERE';
    void AddInLinearContribution(const typename TFeature::PreProcessType& x, const SystemVectorRef& l) const
    {
	AddInLinearContributionAboveDiagonal(x, l);
	AddInLinearContributionBelowDiagonal(x, l);
    }

HERE
    
    AddInLinearContributionAboveDiagonal();
    print STDOUT "\n";
    AddInLinearContributionBelowDiagonal();
}

sub AddInLinearContributionAboveDiagonal
{
    print STDOUT 'void AddInLinearContributionAboveDiagonal(const typename TFeature::PreProcessType& x, const SystemVectorRef& l) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = l.NumPixels();
const auto scale = TValue(1.0)/N;
const auto tail = w.Wl.template tail < VarDim > ();

HERE

    Declarations();

    print STDOUT <<'HERE1';
for( int idx = 1; idx < N; ++idx ) { 
const auto i  = idx, j = idx-1;
HERE1

    InitializeFactors();
    LinearUpdates("tail");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub AddInLinearContributionBelowDiagonal
{
    print STDOUT 'void AddInLinearContributionBelowDiagonal(const typename TFeature::PreProcessType& x, const SystemVectorRef& l) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = l.NumPixels();
const auto scale = TValue(1.0)/N;
const auto head = w.Wl.template head < VarDim > ();

HERE

    Declarations();

    print STDOUT <<'HERE1';
for( int idx = N-2; idx >= 0; --idx ) { 
const auto i  = idx, j = idx+1;
HERE1

    InitializeFactors();
    LinearUpdates("head");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub AccumulateLinearGradientContributionAboveDiagonal
{
    print STDOUT 'void AccumulateLinearGradientContributionAboveDiagonal(const typename TFeature::PreProcessType& x, const SystemVectorCRef& y, const SystemVectorCRef& c, TValue normC) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = y.NumPixels();
const auto scale = TValue(1.0)/(N*normC);
auto& Gtail = w.Gl.template tail < VarDim > ();

HERE

    Declarations();

    print STDOUT <<'HERE1';
for( int idx = 1; idx < N; ++idx ) { 
const auto i  = idx, j = idx-1;
HERE1

    InitializeFactors();
    LinearGradientUpdates("Gtail");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub AccumulateLinearGradientContributionBelowDiagonal
{
    print STDOUT 'void AccumulateLinearGradientContributionBelowDiagonal(const typename TFeature::PreProcessType& x, const SystemVectorCRef& y, const SystemVectorCRef& c, TValue normC) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = y.NumPixels();
const auto scale = TValue(1.0)/(N*normC);
auto& Ghead = w.Gl.template head < VarDim > ();

HERE

    Declarations();

    print STDOUT <<'HERE1';
for( int idx = N-2; idx >= 0; --idx ) { 
const auto i  = idx, j = idx+1;
HERE1

    InitializeFactors();
    LinearGradientUpdates("Ghead");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub AccumulateGradient
{
print STDOUT <<'HERE';
    void AccumulateGradient(const typename TFeature::PreProcessType& x,
			    const SystemVectorCRef& y, const SystemVectorCRef& c, TValue normC) const
{
    AccumulateQuadraticGradientContributionAboveDiagonal(x, y, c, normC);
    AccumulateQuadraticGradientContributionBelowDiagonal(x, y, c, normC);
    AccumulateLinearGradientContributionAboveDiagonal(x, y, c, normC);
    AccumulateLinearGradientContributionBelowDiagonal(x, y, c, normC);
}

HERE
    AccumulateQuadraticGradientContributionAboveDiagonal();
print STDOUT "\n";
    AccumulateQuadraticGradientContributionBelowDiagonal();
print STDOUT "\n";
    AccumulateLinearGradientContributionAboveDiagonal();
print STDOUT "\n";
    AccumulateLinearGradientContributionBelowDiagonal();    
}

sub AddInImplicitMatrixMultipliedBy
{
    print STDOUT <<'HERE';
    void AddInImplicitMatrixMultipliedBy(const typename TFeature::PreProcessType& x,
					  const SystemVectorRef& Fy, const SystemVectorCRef& y) const
    {
	AddInImplicitMatrixAboveDiagonalMultipliedBy(x, Fy, y);
	AddInImplicitMatrixBelowDiagonalMultipliedBy(x, Fy, y);
    }

HERE
    AddInImplicitMatrixAboveDiagonalMultipliedBy();
    print STDOUT "\n";
    AddInImplicitMatrixBelowDiagonalMultipliedBy();
}

sub AddInImplicitMatrixAboveDiagonalMultipliedBy
{
    print STDOUT 'void AddInImplicitMatrixAboveDiagonalMultipliedBy(const typename TFeature::PreProcessType& x, const SystemVectorRef& Fy, const SystemVectorCRef& y) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = y.NumPixels();
const auto scale = TValue(1.0)/N;

const auto varT = - w.Wq.template bottomRightCorner < VarDim, VarDim > ();
const auto covT = - w.Wq.template bottomLeftCorner < VarDim, VarDim > ();

HERE

    Declarations(1);

    print STDOUT <<'HERE1';
for( int idx = 1; idx < N; ++idx ) { 
const auto i  = idx, j = idx-1;
HERE1

    InitializeFactors();
    QuadraticUpdates("covT", "varT");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub AddInImplicitMatrixBelowDiagonalMultipliedBy
{
    print STDOUT 'void AddInImplicitMatrixBelowDiagonalMultipliedBy(const typename TFeature::PreProcessType& x, const SystemVectorRef& Fy, const SystemVectorCRef& y) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = y.NumPixels();
const auto scale = TValue(1.0)/N;

const auto var  = - w.Wq.template topLeftCorner < VarDim, VarDim > ();
const auto cov  = - w.Wq.template topRightCorner < VarDim, VarDim > ();

HERE

    Declarations(1);

    print STDOUT <<'HERE1';
for( int idx = N-2; idx >= 0; --idx ) { 
const auto i  = idx, j = idx+1;
HERE1

    InitializeFactors();
    QuadraticUpdates("cov", "var");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub AccumulateQuadraticGradientContributionAboveDiagonal
{
    print STDOUT 'void AccumulateQuadraticGradientContributionAboveDiagonal(const typename TFeature::PreProcessType& x, const SystemVectorCRef& y, const SystemVectorCRef& c, TValue normC) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = y.NumPixels();
const auto scale = TValue(1.0)/(N*normC);

auto& GvarT = w.Gq.template bottomRightCorner < VarDim, VarDim > ();
auto& GcovT = w.Gq.template bottomLeftCorner < VarDim, VarDim > ();
HERE

    Declarations(1);

    print STDOUT <<'HERE1';
for( int idx = 1; idx < N; ++idx ) { 
const auto i  = idx, j = idx-1;
HERE1

    InitializeFactors();
    QuadraticGradientUpdates("GcovT", "GvarT");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub AccumulateQuadraticGradientContributionBelowDiagonal
{
    print STDOUT 'void AccumulateQuadraticGradientContributionBelowDiagonal(const typename TFeature::PreProcessType& x, const SystemVectorCRef& y, const SystemVectorCRef& c, TValue normC) const' . "\n";
    print STDOUT '{' . "\n";

    print STDOUT <<'HERE';
const auto N = y.NumPixels();
const auto scale = TValue(1.0)/(N*normC);

auto& Gvar  = w.Gq.template topLeftCorner < VarDim, VarDim > ();
auto& Gcov  = w.Gq.template topRightCorner < VarDim, VarDim > ();

HERE

    Declarations(1);

    print STDOUT <<'HERE1';
for( int idx = N-2; idx >= 0; --idx ) { 
const auto i  = idx, j = idx+1;
HERE1

    InitializeFactors();
    QuadraticGradientUpdates("Gcov", "Gvar");

print STDOUT '}' . "\n";


    print STDOUT '}' . "\n";
}

sub BoilerPlate
{
    print STDOUT 'template<typename TFeature, typename TUnaryGroundLabel>' . "\n";
    print STDOUT "class ${Name}Operator : public OperatorBase<TFeature, TUnaryGroundLabel>" . "\n";
    print STDOUT <<HERE0;	
    {
      public:
	static const size_t VarDim = TUnaryGroundLabel::Size;
	typedef typename TUnaryGroundLabel::ValueType TValue;

	typedef Compute::SystemVectorRef<TValue, VarDim>  SystemVectorRef;
	typedef Compute::SystemVectorCRef<TValue, VarDim> SystemVectorCRef;

	typedef Eigen::Matrix<TValue, VarDim, VarDim>   TVarVarMatrix;
	typedef Eigen::Matrix<TValue, VarDim, 1>        TVarVector;
	typedef Eigen::Matrix<TValue, -1, 1>            TVector;

	typedef Compute::Weights<TValue, 2*VarDim, 1> TWeights;

	size_t f;
	size_t p;
	TWeights w;
	TValue smallestEigenvalue;
	TValue largestEigenvalue;

	${Name}Operator() : f(0), p(0), w(1e-6), smallestEigenvalue(1e-6), largestEigenvalue(1e2) {}
	${Name}Operator(size_t quadraticBasisIndex1, size_t quadraticBasisIndex2, TValue smallestEigenvalue_, TValue largestEigenvalue_) 
	    : f(quadraticBasisIndex1), p(quadraticBasisIndex2), w(smallestEigenvalue_), smallestEigenvalue(smallestEigenvalue_), largestEigenvalue(largestEigenvalue_) {}

	static OperatorRef<TFeature, TUnaryGroundLabel> Instantiate()
	{
	    return OperatorRef<TFeature, TUnaryGroundLabel>(new ${Name}Operator<TFeature, TUnaryGroundLabel>());
	}

	static OperatorRef<TFeature, TUnaryGroundLabel> Instantiate(size_t quadraticBasisIndex1, size_t quadraticBasisIndex2, TValue smallestEigenvalue_, TValue largestEigenvalue_)
	{
	    return OperatorRef<TFeature, TUnaryGroundLabel>(new ${Name}Operator<TFeature, TUnaryGroundLabel>(quadraticBasisIndex1, quadraticBasisIndex2,
															       smallestEigenvalue_,
															       largestEigenvalue_));
	}
HERE0

    print STDOUT <<'HERE1';

    // Check if W is element-wise box-constrained and symmetric
        const TValue* CheckFeasibility(const TValue *ws, bool& feasible) const
    {
	return TWeights::CheckFeasibility(ws, feasible);
    }

    // Project onto the convex set of symmetric, element-wise bounded matrices
        TValue* Project(TValue *ws) const 
    {
	//return TWeights::ProjectDiagForm(ws, smallestEigenvalue, largestEigenvalue);
	return TWeights::Project(ws, smallestEigenvalue, largestEigenvalue);
    }

    // One weight per label combination + one weight per label for the linear part
        size_t NumWeights() const 
    {
	return TWeights::NumWeights();
    }
        
        const TValue* SetWeights(const TValue *ws) 
	{
	    return w.SetWeights(ws);
	}
        
        TValue* GetWeights(TValue *ws) const
	{
	    return w.GetWeights(ws);
	}
        
    // No prior for now, but Frobenius norm seems sensible, for instance
        TValue* GetGradientAddPrior(TValue *gs, TValue& objective) const 
    {
	return w.GetGradient(gs);
    }
        
        void ClearGradient() 
	{
	    w.ClearGradient();
	}

    void ResetWeights()
    {
	w.Reset(smallestEigenvalue);
    }

    void Print() const 
    {
    }        

    OperatorType Type() const
    {
HERE1

    print STDOUT "return ${Name};\n";

print STDOUT <<'HERE2';
  }

  std::istream& Deserialize(std::istream& in)
  {
      in >> smallestEigenvalue;
      in >> largestEigenvalue;
      in >> f;
      in >> p;
      in >> w;
      return in;
  }

  std::ostream& Serialize(std::ostream& out) const
  {
      out << smallestEigenvalue << " " << largestEigenvalue << std::endl;
      out << f << std::endl;
      out << p << std::endl;
      out << w << std::endl;
      return out;
  }

#ifdef _OPENMP
    // We don't need a lock
        void InitializeLocks() 
{
}
        
void DestroyLocks() 
{
}
#endif

HERE2

}


exit 0;
